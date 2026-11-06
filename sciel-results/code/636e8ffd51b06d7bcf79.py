import time
import random

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 50.0  # Safety margin from 60s limit

# Pre-compute masks at module load time to avoid doing it in construct()
_INITIAL_MASKS = [0] * N
for _w in range(N):
    block = 1 << _w
    pattern = (1 << block) - 1
    pattern <<= block
    num_periods = (1 << N) // (2 * block)
    m = 0
    shift = 0
    for _ in range(num_periods):
        m |= (pattern << shift)
        shift += 2 * block
    _INITIAL_MASKS[_w] = m

def _odd_even_transposition(n=N):
    """Return the odd-even transposition sorting network for n wires."""
    return [(i, i + 1) for phase in range(n)
            for i in range(phase % 2, n - 1, 2)]

def _verify(comparators):
    """
    Verify if the comparator list sorts all 2**N boolean inputs.
    Uses bitwise operations on Python integers representing the state of each wire.
    """
    if not comparators:
        return N <= 1
    
    state = _INITIAL_MASKS[:]
    
    for i, j in comparators:
        a = state[i]
        b = state[j]
        state[i] = a & b
        state[j] = a | b
        
    for i in range(N - 1):
        if (state[i] & ~state[i+1]) != 0:
            return False
            
    return True

def _greedy_cascade(current, deadline, max_steps=10):
    """
    After a successful mutation, try to remove more comparators greedily.
    This helps escape shallow local minima immediately after a good move.
    """
    for _ in range(max_steps):
        if time.time() > deadline:
            break
        if len(current) <= 70:
            break
        
        # Try removing a random comparator
        idx = random.randint(0, len(current) - 1)
        candidate = current[:idx] + current[idx + 1:]
        
        if _verify(candidate):
            current = candidate
        else:
            break
            
    return current

def _build_initial_network():
    """
    Build a good initial sorting network.
    Start with odd-even transposition and apply an initial greedy reduction pass.
    """
    network = _odd_even_transposition()
    
    # Quick greedy pass to remove obvious redundancies
    t0 = time.time()
    changed = True
    while changed and time.time() - t0 < 3.0:
        changed = False
        for i in range(len(network) - 1, -1, -1):
            neighbor = network[:i] + network[i+1:]
            if _verify(neighbor):
                network = neighbor
                changed = True
                break  # Restart from beginning after a change
    
    return network

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS

    # Build initial network
    current = _build_initial_network()
    best = current[:]
    best_len = len(best)
    
    # Initial verification
    if not _verify(best):
        current = _odd_even_transposition()
        best = current[:]
        best_len = len(best)
    
    rng = random.Random(42)
    
    # Phase 1: Aggressive greedy removal
    # Try to remove each comparator one by one. If the network is still valid, keep it removed.
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(current) - 1, -1, -1):
            if time.time() > deadline:
                break
            # Try removing current[i]
            neighbor = current[:i] + current[i+1:]
            if _verify(neighbor):
                current = neighbor
                changed = True
                if len(current) < best_len:
                    best = current[:]
                    best_len = len(current)
                break  # Restart from beginning after a change
    
    # Phase 2: Simulated Annealing with diverse moves
    # Key insight: diverse moves (remove, swap, replace, batch_remove) allow escaping local minima
    T = 8.0
    T_min = 0.05
    cooling_rate = 0.995
    
    all_comparators = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Track consecutive failures to adjust temperature
    fail_count = 0
    success_count = 0
    consecutive_failures = 0
    max_consecutive_failures = 3000
    
    while time.time() < deadline and T > T_min:
        if len(current) <= 70:
            break
            
        if consecutive_failures > max_consecutive_failures:
            # Restart with fresh random seed from best
            rng = random.Random(rng.randint(0, 2**32 - 1))
            consecutive_failures = 0
            current = best[:]
            T = 8.0  # Reset temperature on restart
        
        move_type = rng.choices(
            ['remove', 'swap', 'replace', 'batch_remove'],
            weights=[5, 2, 2, 1]
        )[0]
        
        neighbor = current[:]
        valid_move = True
        
        if move_type == 'remove' and len(neighbor) > 0:
            idx = rng.randrange(len(neighbor))
            neighbor.pop(idx)
            
        elif move_type == 'swap' and len(neighbor) >= 2:
            i = rng.randrange(len(neighbor))
            j = rng.randrange(len(neighbor))
            if i == j:
                valid_move = False
            else:
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                
        elif move_type == 'replace' and len(neighbor) > 0:
            idx = rng.randrange(len(neighbor))
            # Generate random valid comparator, biased towards short distances
            if rng.random() < 0.5:
                dist = rng.randint(1, 3)
            else:
                dist = rng.randint(1, N - 1)
            a = rng.randint(0, N - 1 - dist)
            b = a + dist
            neighbor[idx] = (a, b)
            
        elif move_type == 'batch_remove' and len(neighbor) >= 3:
            num_remove = rng.randint(2, 3)
            if len(neighbor) <= num_remove:
                valid_move = False
            else:
                remove_indices = set(rng.sample(range(len(neighbor)), num_remove))
                neighbor = [c for idx, c in enumerate(neighbor) if idx not in remove_indices]
        
        if not valid_move:
            continue
            
        if _verify(neighbor):
            delta = len(neighbor) - len(current)
            # Accept if better or with probability exp(-delta/T)
            if delta <= 0 or rng.random() < 2.718281828459045 ** (-delta / T):
                current = neighbor
                if len(current) < best_len:
                    best = current[:]
                    best_len = len(current)
                success_count += 1
                fail_count = 0
                consecutive_failures = 0
                
                # Greedy cascade after successful move
                current = _greedy_cascade(current, deadline, max_steps=5)
                if len(current) < best_len:
                    best = current[:]
                    best_len = len(current)
            else:
                fail_count += 1
                consecutive_failures += 1
        else:
            fail_count += 1
            consecutive_failures += 1
        
        # Adjust cooling rate based on success
        if success_count > 10:
            cooling_rate = 0.998  # Cool slower when making progress
            success_count = 0
        elif fail_count > 50:
            cooling_rate = 0.99  # Cool faster when stuck
            fail_count = 0
            
        T *= cooling_rate
        
        # Periodic restart: if stuck for too long, try a fresh greedy pass
        if time.time() < deadline and len(current) > best_len + 5:
            # Try to get back to best by greedy removal from best
            candidate = best[:]
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for i in range(len(candidate) - 1, -1, -1):
                    neighbor = candidate[:i] + candidate[i+1:]
                    if _verify(neighbor):
                        candidate = neighbor
                        changed = True
                        break
            if len(candidate) < len(current):
                current = candidate

    # Final check
    if not _verify(best):
        return _odd_even_transposition()
        
    return best
