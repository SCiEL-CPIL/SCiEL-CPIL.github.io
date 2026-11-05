import time
from fractions import Fraction
import itertools

def _sieve(n):
    if n < 2:
        return []
    is_prime = bytearray([1]) * (n + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, n + 1) if is_prime[i]]

def _compute_L(covered, max_check=100000):
    L = 0
    for i in range(1, min(max_check + 1, len(covered))):
        if covered[i]:
            L = i
        else:
            break
    return L

def _greedy_assign(P_sorted, max_L=100000):
    """Greedy: for n=1..max_L, if not covered, assign smallest unused prime."""
    covered = [False] * (max_L + 1)
    residues = {}
    used_primes = set()
    
    for n in range(1, max_L + 1):
        if covered[n]:
            continue
        assigned = False
        for p in P_sorted:
            if p in used_primes:
                continue
            r = n % p
            residues[p] = r
            used_primes.add(p)
            start_k = r if r > 0 else p
            for k in range(start_k, max_L + 1, p):
                covered[k] = True
            assigned = True
            break
        if not assigned:
            break
    
    L = _compute_L(covered, max_L)
    classes = [[p, residues[p]] for p in P_sorted if p in residues]
    return L, classes, covered

def _local_search(classes, P_sorted, max_L=100000, deadline=None):
    """Try swapping residues to improve L."""
    residues = {p: r for p, r in classes}
    used_primes = set(residues.keys())
    
    # Compute initial covered
    covered = [False] * (max_L + 1)
    for p, r in classes:
        start_k = r if r > 0 else p
        for k in range(start_k, max_L + 1, p):
            covered[k] = True
    
    current_L = _compute_L(covered, max_L)
    best_L = current_L
    best_classes = [list(c) for c in classes]
    
    # Try changing residues for each prime to improve coverage at the gap
    # The gap is at current_L + 1
    gap = current_L + 1
    
    # For each prime, try all residues and see if it helps cover the gap
    # or extends the prefix
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        gap = best_L + 1
        if gap > max_L:
            break
        
        # Recompute covered from best_classes
        covered = [False] * (max_L + 1)
        for p, r in best_classes:
            start_k = r if r > 0 else p
            for k in range(start_k, max_L + 1, p):
                covered[k] = True
        
        # For each prime, try to change its residue to cover the gap
        for idx, (p, r) in enumerate(best_classes):
            if deadline is not None and time.time() > deadline:
                break
            # Try all residues
            for new_r in range(p):
                if new_r == r:
                    continue
                # Check if this residue helps cover the gap
                if gap % p != new_r:
                    # This residue doesn't cover the gap directly,
                    # but might help cover other gaps further along
                    # Skip for efficiency - only try residues that cover gap
                    continue
                
                # Tentatively change
                old_r = best_classes[idx][1]
                best_classes[idx][1] = new_r
                
                # Recompute covered
                new_covered = [False] * (max_L + 1)
                for pp, rr in best_classes:
                    start_k = rr if rr > 0 else pp
                    for k in range(start_k, max_L + 1, pp):
                        new_covered[k] = True
                
                new_L = _compute_L(new_covered, max_L)
                
                if new_L > best_L:
                    best_L = new_L
                    improved = True
                    # Keep this change
                    break
                else:
                    # Revert
                    best_classes[idx][1] = old_r
        
        # Also try adding a new prime if budget allows
        if not improved:
            # Check if we can add a prime to cover the gap
            current_sum = sum(Fraction(1, p) for p, r in best_classes)
            for p in P_sorted:
                if p in used_primes:
                    continue
                if current_sum + Fraction(1, p) > 1:
                    continue
                if gap % p != 0 and gap % p != gap % p:  # always true, just checking
                    pass
                # Try adding p with residue gap % p
                new_r = gap % p
                test_classes = [list(c) for c in best_classes] + [[p, new_r]]
                
                new_covered = [False] * (max_L + 1)
                for pp, rr in test_classes:
                    start_k = rr if rr > 0 else pp
                    for k in range(start_k, max_L + 1, pp):
                        new_covered[k] = True
                
                new_L = _compute_L(new_covered, max_L)
                if new_L > best_L:
                    best_L = new_L
                    best_classes = test_classes
                    used_primes.add(p)
                    current_sum += Fraction(1, p)
                    improved = True
                    break
    
    return best_L, best_classes

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    primes = _sieve(1000)
    small_primes = [2, 3, 5, 7, 11]
    large_primes = [p for p in primes if p > 11]
    
    # Generate candidate prime sets
    candidate_sets = []
    seen_sets = set()
    
    # Seed: primes >= 13
    seed_set = [p for p in primes if p >= 13]
    candidate_sets.append(seed_set)
    seen_sets.add(tuple(sorted(seed_set)))
    
    # Systematic: subsets of small primes + fill with large primes
    for r in range(len(small_primes) + 1):
        for subset in itertools.combinations(small_primes, r):
            current_sum = Fraction(0)
            for p in subset:
                current_sum += Fraction(1, p)
            if current_sum > 1:
                continue
            
            current_set = list(subset)
            for p in large_primes:
                if current_sum + Fraction(1, p) <= 1:
                    current_set.append(p)
                    current_sum += Fraction(1, p)
            
            key = tuple(sorted(current_set))
            if key not in seen_sets:
                seen_sets.add(key)
                candidate_sets.append(current_set)
    
    # Targeted swaps: seed minus some primes plus smaller ones
    swap_candidates = [
        ([53, 59], [11]),
        ([13], [7]),
        ([13, 17], [5]),
        ([13, 17, 19, 23, 29], [3]),
        ([p for p in seed_set if p <= 47], [2]),
        ([13, 53, 59], [7, 11]),
        ([13, 17, 53, 59], [5, 11]),
        ([13, 17, 19], [7, 11]),
        ([13, 17, 19, 23], [5, 7]),
        ([13, 17, 19, 23, 29], [5, 7, 11]),
    ]
    for to_remove, to_add in swap_candidates:
        new_set = [p for p in seed_set if p not in to_remove]
        current_sum = sum(Fraction(1, p) for p in new_set)
        valid_add = []
        for p in to_add:
            if p in new_set:
                continue
            if current_sum + Fraction(1, p) <= 1:
                valid_add.append(p)
                current_sum += Fraction(1, p)
        new_set.extend(valid_add)
        key = tuple(sorted(new_set))
        if key not in seen_sets:
            seen_sets.add(key)
            candidate_sets.append(new_set)
    
    # Evaluate each candidate
    best_L = 0
    best_classes = []
    
    # First pass: quick greedy on all candidates
    for P in candidate_sets:
        if time.time() > deadline - 30:
            break
        P_sorted = sorted(P)
        L, classes, _ = _greedy_assign(P_sorted, max_L=100000)
        if L > best_L:
            best_L = L
            best_classes = classes
    
    # Second pass: local search on top candidates
    # Sort candidates by their greedy L
    candidate_scores = []
    for P in candidate_sets:
        if time.time() > deadline - 20:
            break
        P_sorted = sorted(P)
        L, classes, _ = _greedy_assign(P_sorted, max_L=100000)
        candidate_scores.append((L, P_sorted, classes))
    
    candidate_scores.sort(key=lambda x: -x[0])
    
    # Local search on top 20 candidates
    for L, P_sorted, classes in candidate_scores[:20]:
        if time.time() > deadline - 10:
            break
        new_L, new_classes = _local_search(classes, P_sorted, max_L=100000, deadline=deadline - 5)
        if new_L > best_L:
            best_L = new_L
            best_classes = new_classes
    
    # Verify budget
    s = sum(Fraction(1, p) for p, r in best_classes)
    if s > 1:
        # Fallback to seed
        seed_set = [p for p in primes if p >= 13]
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]
    
    return best_classes

if __name__ == "__main__":
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Time: {elapsed:.1f}s, Classes: {len(result)}")
    # Compute L
    max_L = 100000
    covered = [False] * (max_L + 1)
    for p, r in result:
        start_k = r if r > 0 else p
        for k in range(start_k, max_L + 1, p):
            covered[k] = True
    L = 0
    for i in range(1, max_L + 1):
        if covered[i]:
            L = i
        else:
            break
    print(f"L = {L}")
    s = sum(Fraction(1, p) for p, r in result)
    print(f"Sum 1/p = {float(s):.6f}")
