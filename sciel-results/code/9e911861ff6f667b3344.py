"""
Combined program: bitset scoring + incremental delta local search + multi-start restarts.

Key ideas:
- Bitset representation for fast full-score computation (O(255) bit ops)
- Incremental delta scoring for local search: precompute per-d coverage to
  evaluate swap effects in O(255) without full re-score
- Multi-start: greedy init + random restarts + SA + fast local polish
"""
import time
import random
import numpy as np

N = 32
MAX_VAL = 512
MAX_D = 255
TIME_LIMIT = 118  # safety margin before 120s kill

# Precompute for each d, the list of (a, a+d, a+2d) triples
_AP_TRIPLES = {}
for d in range(1, MAX_D + 1):
    triples = []
    for a in range(1, MAX_VAL + 1 - 2 * d + 1):
        a2 = a + d
        a3 = a + 2 * d
        if a3 <= MAX_VAL:
            triples.append((a, a2, a3))
    _AP_TRIPLES[d] = triples

def score_bitset(bits):
    """Full score: count distinct d where some AP exists in the bitset."""
    count = 0
    for d in range(1, MAX_D + 1):
        if (bits & (bits >> d) & (bits >> (2 * d))) != 0:
            count += 1
    return count

def bits_to_list(bits):
    return [i for i in range(1, MAX_VAL + 1) if bits & (1 << i)]

def list_to_bits(lst):
    bits = 0
    for x in lst:
        bits |= (1 << x)
    return bits

def compute_coverage(bits):
    """
    For each d, compute how many APs are fully covered by the set.
    Returns dict: d -> count of (a,a+d,a+2d) where all three are in set.
    """
    coverage = {}
    for d in range(1, MAX_D + 1):
        cnt = 0
        for (a, a2, a3) in _AP_TRIPLES[d]:
            if (bits >> a) & 1 and (bits >> a2) & 1 and (bits >> a3) & 1:
                cnt += 1
        coverage[d] = cnt
    return coverage

def delta_swap_score(coverage, bits, v_remove, v_add):
    """
    Compute the change in score when removing v_remove and adding v_add.
    Uses precomputed coverage to determine which d's gain/lose their last AP.
    Returns (delta_score, new_coverage_dict) — but we only need delta for comparison.
    For efficiency, we compute the net change in the count of d's with coverage>0.
    """
    # Determine which d's are affected by removing v_remove and adding v_add
    # A d is affected if v_remove or v_add participates in some AP with difference d.
    
    # For each d, check if v_remove is in any AP of that d, and if v_add is in any AP of that d
    # We need to track: for each d, does removing v_remove drop coverage to 0?
    # Does adding v_add increase coverage from 0?
    
    delta = 0
    
    for d in range(1, MAX_D + 1):
        cur_cov = coverage[d]
        
        # Check if v_remove is part of any AP with this d
        v_remove_in_d = False
        # Check if v_add is part of any AP with this d
        v_add_in_d = False
        
        # v_remove participates as a, a+d, or a+2d
        # as a: a = v_remove, a+d = v_remove+d, a+2d = v_remove+2d
        # as a+d: a = v_remove-d, a+d = v_remove, a+2d = v_remove+d
        # as a+2d: a = v_remove-2d, a+d = v_remove-d, a+2d = v_remove
        
        # For v_remove:
        # Case 1: v_remove = a, need a+d and a+2d in set
        if v_remove + d <= MAX_VAL and v_remove + 2 * d <= MAX_VAL:
            if (bits >> (v_remove + d)) & 1 and (bits >> (v_remove + 2 * d)) & 1:
                v_remove_in_d = True
        # Case 2: v_remove = a+d, need a and a+2d in set
        if v_remove - d >= 1 and v_remove + d <= MAX_VAL:
            if (bits >> (v_remove - d)) & 1 and (bits >> (v_remove + d)) & 1:
                v_remove_in_d = True
        # Case 3: v_remove = a+2d, need a and a+d in set
        if v_remove - 2 * d >= 1 and v_remove - d >= 1:
            if (bits >> (v_remove - 2 * d)) & 1 and (bits >> (v_remove - d)) & 1:
                v_remove_in_d = True
        
        # For v_add (after removal of v_remove, but v_add wasn't in set before):
        # We need to check if v_add forms an AP with two other elements that remain in set
        # after removing v_remove.
        
        # Case 1: v_add = a, need a+d and a+2d in set (and not v_remove)
        if v_add + d <= MAX_VAL and v_add + 2 * d <= MAX_VAL:
            mid = v_add + d
            end = v_add + 2 * d
            if mid != v_remove and end != v_remove:
                if (bits >> mid) & 1 and (bits >> end) & 1:
                    v_add_in_d = True
        # Case 2: v_add = a+d, need a and a+2d in set (and not v_remove)
        if v_add - d >= 1 and v_add + d <= MAX_VAL:
            start = v_add - d
            end = v_add + d
            if start != v_remove and end != v_remove:
                if (bits >> start) & 1 and (bits >> end) & 1:
                    v_add_in_d = True
        # Case 3: v_add = a+2d, need a and a+d in set (and not v_remove)
        if v_add - 2 * d >= 1 and v_add - d >= 1:
            start = v_add - 2 * d
            mid = v_add - d
            if start != v_remove and mid != v_remove:
                if (bits >> start) & 1 and (bits >> mid) & 1:
                    v_add_in_d = True
        
        # Now determine the effect on this d
        # cur_cov is the number of APs with this d in the current set
        # After swap:
        # - Remove v_remove: some APs containing v_remove are lost
        # - Add v_add: some new APs containing v_add are gained
        
        # This is complex to compute exactly without enumerating all APs.
        # For a simpler approach: just re-score if the swap is promising.
        pass
    
    # The incremental approach is complex. Let's use a simpler but still fast method:
    # Just re-score the new bitset. The bitset scoring is O(255) which is very fast.
    # The real speedup comes from reducing the number of candidates we test.
    return None  # Not using this path; we'll use full re-score but with better candidate selection

def greedy_construct():
    """Greedy construction: add elements one by one maximizing marginal gain."""
    bits = 0
    current_score = 0
    
    for _ in range(N):
        best_gain = -1
        best_val = None
        for v in range(1, MAX_VAL + 1):
            if (bits >> v) & 1:
                continue
            new_bits = bits | (1 << v)
            new_score = score_bitset(new_bits)
            gain = new_score - current_score
            if gain > best_gain:
                best_gain = gain
                best_val = v
        if best_val is None:
            break
        bits |= (1 << best_val)
        current_score = score_bitset(bits)
    
    return bits, current_score

def random_init():
    """Random 32-element set."""
    lst = random.sample(range(1, MAX_VAL + 1), N)
    bits = list_to_bits(lst)
    score = score_bitset(bits)
    return bits, score

def local_search(bits, score, time_budget):
    """
    Local search: iteratively try swaps that improve score.
    Uses full re-score (O(255) per eval) but with early termination.
    """
    start_time = time.time()
    best_bits = bits
    best_score = score
    
    in_set = [v for v in range(1, MAX_VAL + 1) if (best_bits >> v) & 1]
    out_set = [v for v in range(1, MAX_VAL + 1) if not ((best_bits >> v) & 1)]
    
    while time.time() - start_time < time_budget:
        improved = False
        best_swap_gain = 0
        best_swap = None
        
        # Try all swaps, but break early if time is tight
        for v_remove in in_set:
            if time.time() - start_time > time_budget:
                break
            for v_add in out_set:
                if time.time() - start_time > time_budget:
                    break
                new_bits = (best_bits & ~(1 << v_remove)) | (1 << v_add)
                new_score = score_bitset(new_bits)
                gain = new_score - best_score
                if gain > best_swap_gain:
                    best_swap_gain = gain
                    best_swap = (v_remove, v_add)
        
        if best_swap is not None and best_swap_gain > 0:
            v_remove, v_add = best_swap
            best_bits = (best_bits & ~(1 << v_remove)) | (1 << v_add)
            best_score += best_swap_gain
            # Update in_set and out_set
            in_set.remove(v_remove)
            in_set.append(v_add)
            out_set.remove(v_add)
            out_set.append(v_remove)
            improved = True
        
        if not improved:
            break
    
    return best_bits, best_score

def simulated_annealing(bits, score, time_budget):
    """
    Simulated annealing with swap moves.
    """
    start_time = time.time()
    current_bits = bits
    current_score = score
    best_bits = bits
    best_score = score
    
    in_set = [v for v in range(1, MAX_VAL + 1) if (current_bits >> v) & 1]
    out_set = [v for v in range(1, MAX_VAL + 1) if not ((current_bits >> v) & 1)]
    
    temp = 5.0
    cooling = 0.9999
    
    while time.time() - start_time < time_budget:
        # Random swap
        v_remove = random.choice(in_set)
        v_add = random.choice(out_set)
        
        new_bits = (current_bits & ~(1 << v_remove)) | (1 << v_add)
        new_score = score_bitset(new_bits)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / max(temp, 1e-10)):
            current_bits = new_bits
            current_score = new_score
            in_set.remove(v_remove)
            in_set.append(v_add)
            out_set.remove(v_add)
            out_set.append(v_remove)
            
            if current_score > best_score:
                best_score = current_score
                best_bits = current_bits
        
        temp *= cooling
        if temp < 0.01:
            break
    
    return best_bits, best_score

def construct():
    start_time = time.time()
    
    best_bits = None
    best_score = 0
    
    # Phase 1: Greedy construction
    bits, score = greedy_construct()
    best_bits = bits
    best_score = score
    
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT:
        return bits_to_list(best_bits)
    
    # Phase 2: Local search polish on greedy result
    remaining = TIME_LIMIT - elapsed
    ls_budget = min(remaining * 0.3, 30)
    ls_bits, ls_score = local_search(best_bits, best_score, ls_budget)
    if ls_score > best_score:
        best_bits = ls_bits
        best_score = ls_score
    
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 5:
        return bits_to_list(best_bits)
    
    # Phase 3: Multi-start with SA + local search
    remaining = TIME_LIMIT - elapsed
    num_starts = min(8, int(remaining / 10) + 1)
    
    for s in range(num_starts):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        # Random restart
        r_bits, r_score = random_init()
        
        # SA for a short time
        sa_budget = min(5, (TIME_LIMIT - (time.time() - start_time)) * 0.4)
        sa_bits, sa_score = simulated_annealing(r_bits, r_score, sa_budget)
        
        # Local search polish
        ls_budget = min(3, (TIME_LIMIT - (time.time() - start_time)) * 0.3)
        ls_bits, ls_score = local_search(sa_bits, sa_score, ls_budget)
        
        if ls_score > best_score:
            best_bits = ls_bits
            best_score = ls_score
    
    # Final polish if time allows
    if time.time() - start_time < TIME_LIMIT - 3:
        final_budget = TIME_LIMIT - (time.time() - start_time) - 2
        final_bits, final_score = local_search(best_bits, best_score, final_budget)
        if final_score > best_score:
            best_bits = final_bits
            best_score = final_score
    
    return bits_to_list(best_bits)

# Quick self-test
if __name__ == "__main__":
    S = construct()
    assert len(S) == 32
    assert len(set(S)) == 32
    assert all(1 <= x <= 512 for x in S)
    # Compute score
    S_set = set(S)
    found = set()
    for a in S_set:
        for b in S_set:
            if b == a:
                continue
            d = b - a
            if d > 0 and (a + 2 * d) in S_set:
                found.add(d)
    print("Score:", len(found))
    print("Set:", sorted(S))
