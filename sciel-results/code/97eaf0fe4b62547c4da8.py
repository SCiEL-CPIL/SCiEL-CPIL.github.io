<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    rng = random.Random(42)
    
    # Phase 1: Greedy construction
    s, score = _greedy_construct()
    bits = _set_to_bits(s)
    
    best_bits = bits
    best_s = set(s)
    best_score = score
    
    # Phase 2: Local search with restarts
    remaining_phases = 5
    phase_time = (deadline - time.time()) / remaining_phases
    
    for phase in range(remaining_phases):
        phase_deadline = min(time.time() + phase_time, deadline)
        
        # Perturb: randomly swap a few elements
        perturbed_s = set(s)
        for _ in range(rng.randint(2, 5)):
            elems = list(perturbed_s)
            idx = rng.randrange(32)
            old_elem = elems[idx]
            new_elem = rng.randint(1, 512)
            if new_elem not in perturbed_s:
                perturbed_s.discard(old_elem)
                perturbed_s.add(new_elem)
        
        perturbed_bits = _set_to_bits(perturbed_s)
        perturbed_score = _score_bitset(perturbed_bits)
        
        # Local search from perturbed state
        ls_bits, ls_s, ls_score = _local_search(
            perturbed_bits, perturbed_s, perturbed_score, phase_deadline, rng
        )
        
        if ls_score > best_score:
            best_score = ls_score
            best_bits = ls_bits
            best_s = ls_s
            s = ls_s
            bits = ls_bits
        else:
            # Keep the best for next restart
            s = best_s
            bits = best_bits
    
    # Final polish with remaining time
    remaining = deadline - time.time()
    if remaining > 5:
        final_bits, final_s, final_score = _local_search(
            bits, s, best_score, deadline, rng
        )
        if final_score > best_score:
            best_score = final_score
            best_bits = final_bits
            best_s = final_s
    
    return sorted(best_s)
=======
def _simulated_annealing(bits, s, score, deadline, rng):
    """
    Simulated Annealing to maximize the score.
    """
    best_bits = bits
    best_s = set(s)
    best_score = score
    
    # Initial temperature
    # Score range is roughly 0 to 256. Delta is small.
    # We want to accept moves that decrease score by ~1 with prob ~0.5 initially.
    # exp(-1/T) = 0.5 => T = 1/ln(2) ~ 1.44.
    # But we want to explore more broadly. Let's start higher.
    T = 10.0
    alpha = 0.9999 # Cooling factor per step
    min_T = 0.01
    
    elems = list(s)
    
    while time.time() < deadline:
        # Pick a random element to remove
        idx = rng.randrange(32)
        removed = elems[idx]
        
        # Pick a random element to add
        new_elem = rng.randint(1, 512)
        if new_elem == removed:
            continue
        if new_elem in s:
            continue
        
        # Perform swap
        old_bits = bits
        bits = (bits & ~(1 << removed)) | (1 << new_elem)
        s.discard(removed)
        s.add(new_elem)
        elems[idx] = new_elem
        
        new_score = _score_bitset(bits)
        delta = new_score - score
        
        if delta >= 0:
            score = new_score
            if score > best_score:
                best_score = score
                best_bits = bits
                best_s = set(s)
        else:
            # Accept with probability exp(delta / T)
            if T > 0:
                prob = math.exp(delta / T)
                if rng.random() < prob:
                    score = new_score
                else:
                    # Reject: revert
                    bits = old_bits
                    s.discard(new_elem)
                    s.add(removed)
                    elems[idx] = removed
        
        # Cool down
        T *= alpha
        if T < min_T:
            T = min_T
            
    return best_bits, best_s, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for safety
    
    rng = random.Random(42)
    
    # Phase 1: Greedy construction to get a decent start
    s, score = _greedy_construct()
    bits = _set_to_bits(s)
    
    best_bits = bits
    best_s = set(s)
    best_score = score
    
    # Phase 2: Simulated Annealing
    # We run SA for the majority of the time.
    # We might restart SA if we get stuck, but SA itself handles restarts via temperature.
    # To be safe, let's just run one long SA.
    sa_deadline = deadline - 5.0
    
    sa_bits, sa_s, sa_score = _simulated_annealing(bits, s, score, sa_deadline, rng)
    
    if sa_score > best_score:
        best_score = sa_score
        best_bits = sa_bits
        best_s = sa_s
    
    # Phase 3: Final local search polish
    if time.time() < deadline:
        final_bits, final_s, final_score = _local_search(
            best_bits, best_s, best_score, deadline, rng
        )
        if final_score > best_score:
            best_score = final_score
            best_bits = final_bits
            best_s = final_s
    
    return sorted(best_s)
>>>>>>> REPLACE

I need to import `math` for `exp`.

