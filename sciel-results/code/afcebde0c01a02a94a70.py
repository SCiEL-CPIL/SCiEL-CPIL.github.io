"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time

def _can_form_target(target, elements):
    """
    Check if target (Fraction) can be formed by a non-empty subset of reciprocals of elements.
    Returns True if a subset exists.
    """
    if not elements:
        return False
    
    # Quick check: sum of all reciprocals < target => impossible
    # sum(Fraction(1, y) for y in elements) is expensive if done every call.
    # But we can precompute or just do the DFS with pruning.
    # Let's do a DFS with memoization or just simple backtracking with pruning.
    
    # Sort elements by descending reciprocal (i.e., ascending value) to find large subsets faster?
    # Or ascending value (larger reciprocals) to hit the target sum faster?
    # If we want to form a sum S, using larger terms first allows us to prune if acc + sum(remaining) < target.
    
    elems = sorted(elements, reverse=True) # Largest values first -> smallest reciprocals last?
    # Wait, to form a sum, it's often better to try large terms first?
    # Actually, for subset sum, sorting usually helps pruning.
    # Let's sort by value descending (1/1, 1/2, 1/3...).
    # No, 1/1 is large. 1/24 is small.
    # If target is 1/2, and we have 1/3, 1/6, 1/6 (not distinct), etc.
    
    # Let's just use a set of achievable sums?
    # The number of subsets is 2^k. For k up to ~15-20, this might be too slow if done naively.
    # But we have strong pruning:
    # 1. If acc > target: prune (reciprocals are positive).
    # 2. If acc + sum(remaining_reciprocals) < target: prune.
    
    # To make pruning (2) work, we need suffix sums of reciprocals.
    
    n = len(elems)
    if n == 0:
        return False
        
    # Precompute suffix sums of reciprocals
    # suffix_sum[i] = sum(1/x for x in elems[i:])
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + Fraction(1, elems[i])
        
    # We sort elems in ascending order of value? 
    # If we sort ascending (1, 2, 3...), reciprocals are descending (1, 1/2, 1/3...).
    # This helps prune: if we take 1, we might exceed target immediately if target < 1.
    # If target is small, we want to avoid taking large reciprocals.
    # So sorting elements by value descending (24, 23, ..., 1) means reciprocals are ascending (1/24, 1/23, ..., 1).
    # This is bad for pruning "acc > target" because we start small.
    # Sorting elements by value ascending (1, 2, ...) means reciprocals are descending (1, 1/2, ...).
    # This is good for pruning "acc > target" if target is small.
    
    elems = sorted(elements) # Ascending value => Descending reciprocal
    
    # Recalculate suffix sums for this order
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + Fraction(1, elems[i])
        
    def dfs(idx, acc):
        if acc == target:
            return True
        if idx == n:
            return False
        if acc > target:
            return False
        # If even taking all remaining cannot reach target
        if acc + suffix_sums[idx] < target:
            return False
            
        # Option 1: Skip elems[idx]
        if dfs(idx + 1, acc):
            return True
            
        # Option 2: Take elems[idx]
        if dfs(idx + 1, acc + Fraction(1, elems[idx])):
            return True
            
        return False

    return dfs(0, Fraction(0))

def _is_valid(S):
    """
    Check if S is a valid reciprocal-subsum-free set.
    For every x in S, 1/x cannot be formed by a subset of {1/y for y in S, y != x}.
    """
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        x = S_list[i]
        target = Fraction(1, x)
        # Elements available for summing: all except x
        others = [y for y in S_list if y != x]
        if _can_form_target(target, others):
            return False
    return True

def _find_best_set():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Strategy: Greedy construction + Local Search / Repair
    
    # 1. Start with a known valid set or empty set
    # Greedy: Add elements one by one if they don't break validity.
    # Order: Which order?
    # Small numbers are "dangerous" as targets (1/1, 1/2...) because it's easier to sum reciprocals to get them?
    # Actually, 1/x is large for small x. It's HARDER to sum many small reciprocals to get a large number?
    # No, 1/1 = 1. Sum of 1/2+1/3+1/6 = 1.
    # 1/2 = 0.5. Sum of 1/3+1/6 = 0.5.
    # Large x (small reciprocal) is harder to hit?
    # 1/24 is small. You need many terms to sum to 1/24?
    # Actually, Egyptian fractions converge slowly.
    # It is generally easier to represent large fractions as sums of smaller ones?
    # Or is it easier to represent small fractions?
    # 1/1 is easy. 1/2 is easy.
    # Let's try adding small numbers first? If we add 1, can we ever remove it later?
    # If 1 is in the set, 1/1=1. Can others sum to 1?
    # If we have {2, 3, 6}, 1/2+1/3+1/6=1. So {1, 2, 3, 6} is invalid.
    # If we have {2, 4, 5...}, can they sum to 1?
    
    # Let's try a greedy approach adding elements in descending order of "danger".
    # Or just iterate 1..24 and add if valid.
    
    best_set = []
    best_size = 0
    
    # Try different starting orders?
    # Order 1: 1, 2, ..., 24
    # Order 2: 24, 23, ..., 1
    
    orders = [
        list(range(1, 25)),
        list(range(24, 0, -1)),
        # Maybe interleave?
    ]
    
    for order in orders:
        current_set = []
        for x in order:
            candidate_set = current_set + [x]
            if _is_valid(candidate_set):
                current_set = candidate_set
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
        if best_size >= 15: # Likely close to optimal, don't spend too much time
            # Try to improve with local search?
            pass
        
        if time.time() > deadline - 5.0:
            break

    # Local Search / Hill Climbing to improve best_set
    # Try swapping elements in/out?
    # Or removing elements to allow adding others?
    
    # Simpler: Check all single additions to best_set from remaining elements
    # And single removals to see if we can add two?
    
    # Let's do a simple local search:
    # While time permits:
    # 1. Try to add any element not in set. If valid, add.
    # 2. Try to remove any element. If it allows adding a different one (net gain?), do it.
    
    current = best_set[:]
    current_set = set(current)
    all_elements = set(range(1, 25))
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        
        # Try adding
        for x in all_elements - current_set:
            test_set = current + [x]
            if _is_valid(test_set):
                current = test_set
                current_set = set(current)
                improved = True
                break # Restart loop with new set
        
        # Try removing and adding
        if not improved:
            # Try removing one element y, and then adding any z not in (current \ {y})
            for y in current:
                reduced = [z for z in current if z != y]
                reduced_set = set(reduced)
                
                # Find any z that can be added to reduced
                found_z = None
                for z in all_elements - reduced_set:
                    if z == y:
                        continue
                    test_set = reduced + [z]
                    if _is_valid(test_set):
                        found_z = z
                        break
                
                if found_z is not None:
                    # Check if this swap actually improves size?
                    # We removed y, added z. Size is same.
                    # But maybe z allows adding more later?
                    # This is a hill climb with equal steps. Risk of cycling.
                    # Let's only swap if we can then add another?
                    # For now, let's skip pure swaps unless we have a specific heuristic.
                    # Or just try to add to the reduced set first.
                    
                    # Actually, if we can add z to reduced, and z != y, we are just moving y to z.
                    # Unless z is "better" (allows more additions), this doesn't help cardinality.
                    # Let's stick to pure additions first.
                    pass

    return current

def construct():
    # Use a global flag to avoid re-computation if called multiple times quickly?
    # The problem says "Each evaluation must finish within 60s".
    # It implies construct() is called once per evaluation.
    
    start_time = time.time()
    
    # Fast fallback: return the seed result if we expect to be slow?
    # The seed returns range(13, 25) which is size 12.
    # We want to beat 12.
    
    # Let's run the heuristic search.
    result = _find_best_set()
    
    # Ensure we return a valid non-empty set
    if not result:
        result = [13]
    
    if not _is_valid(result):
        # Fallback to a known valid set
        # {13, 14, ..., 24} is valid?
        # 1/13 = 1/x + ...?
        # Smallest sum of reciprocals of two distinct elements from {14..24} is 1/14 + 1/15 = 29/210 ~ 0.138.
        # 1/13 ~ 0.077.
        # So 1/13 cannot be formed by sum of reciprocals of larger numbers.
        # Since we only use larger numbers, and they are smaller reciprocals, the sum will be smaller?
        # Wait.
        # If S = {13, ..., 24}.
        # Check x=13. Target 1/13. Others are 14..24.
        # Max sum of reciprocals of ANY subset of others?
        # Sum of 1/14 + 1/15 + ... is finite.
        # But can it equal 1/13?
        # 1/14 < 1/13.
        # 1/14 + 1/15 = 0.138 > 0.077.
        # So it IS possible to exceed.
        # But is it EXACTLY equal?
        # Unlikely for random sets, but possible.
        # However, for the seed range(13, 25), it was considered valid in the seed (score 1.0 implies it was valid? or just the baseline?).
        # The seed program returns list(range(13, 25)) and scores 1.0.
        # The task says "score 1.0 equals the seed program".
        # So range(13, 25) is a valid candidate.
        
        result = list(range(13, 25))
        
    return result
# EVOLVE-BLOCK-END
