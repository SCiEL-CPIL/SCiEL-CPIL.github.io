import time
import random
import sys

def construct():
    """
    Finds a set of distinct integers in 1..256 maximizing the size,
    such that all non-empty subset sums are distinct and no two distinct
    subset sums divide one another.
    """
    start_time = time.time()
    time_limit = 80  # Keep 10s margin for evaluator overhead (90s total)
    
    def compute_sums(S):
        """Compute all subset sums of S (including 0)."""
        sums = {0}
        for x in S:
            sums = sums | {s + x for s in sums}
        return sums
    
    def check_divisibility_fast(sums_set):
        """
        Check if any two distinct non-empty sums in sums_set divide each other.
        Uses boolean array for O(1) membership checks.
        """
        if len(sums_set) <= 1:
            return True
        sorted_sums = sorted(sums_set)
        max_val = sorted_sums[-1]
        present = [False] * (max_val + 1)
        for s in sums_set:
            present[s] = True
        for u in sorted_sums:
            if u == 0:
                continue
            mult = 2
            while u * mult <= max_val:
                if present[u * mult]:
                    return False
                mult += 1
        return True
    
    def can_add_fast(S, x, current_sums):
        """
        Check if we can add x to S.
        Returns (bool, new_sums) if valid.
        """
        # Generate new sums
        new_sums = {x + s for s in current_sums}
        
        # Check distinctness: new_sums should not overlap with current_sums
        if new_sums & current_sums:
            return False, None
        
        # Combined sums
        combined = current_sums | new_sums
        non_empty = combined - {0}
        
        # Check divisibility
        if not check_divisibility_fast(non_empty):
            return False, None
        
        return True, combined
    
    def greedy_build(candidate_order, current_sums=None, S=None):
        """Greedy build using a given order of candidates."""
        if S is None:
            S = []
        if current_sums is None:
            current_sums = {0}
        
        for x in candidate_order:
            if time.time() - start_time > time_limit:
                break
            valid, new_sums = can_add_fast(S, x, current_sums)
            if valid:
                S.append(x)
                current_sums = new_sums
        return S, current_sums
    
    # Strategy: Multiple random restarts with different heuristics
    best_set = []
    best_size = 0
    
    random.seed(12345)
    
    # Heuristic 1: Start with larger numbers (fewer divisibility issues)
    # Heuristic 2: Random order  
    # Heuristic 3: Weighted toward larger numbers
    
    num_restarts = 100
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        if restart < 20:
            # Start with larger numbers first
            candidates = list(range(256, 0, -1))
            random.shuffle(candidates[:100])
            random.shuffle(candidates[100:])
        elif restart < 40:
            # Random order
            candidates = list(range(1, 257))
            random.shuffle(candidates)
        elif restart < 60:
            # Weighted toward larger numbers
            candidates = list(range(1, 257))
            candidates.sort(key=lambda x: -x + random.gauss(0, 40))
        elif restart < 80:
            # Random
            candidates = list(range(1, 257))
            random.shuffle(candidates)
        else:
            # Prefer larger half first
            candidates = list(range(128, 257)) + list(range(1, 128))
            random.shuffle(candidates)
        
        S, _ = greedy_build(candidates)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    # Local search: try to improve best_set by swapping elements
    if best_size >= 3:
        base_set = best_set[:]
        no_improve_count = 0
        max_no_improve = 30
        
        while no_improve_count < max_no_improve:
            if time.time() - start_time > time_limit:
                break
            
            S = base_set[:]
            improved = False
            
            # Try removing one element and adding a better one
            for remove_idx in range(len(S)):
                if time.time() - start_time > time_limit:
                    break
                
                S_no = S[:remove_idx] + S[remove_idx+1:]
                sums_no = compute_sums(S_no)
                
                # Try adding each candidate not in S_no, prefer larger
                for x in range(256, 0, -1):
                    if x in S_no:
                        continue
                    valid, new_sums = can_add_fast(S_no, x, sums_no)
                    if valid:
                        S_new = S_no + [x]
                        if len(S_new) > len(S):
                            S = S_new
                            improved = True
                            break
                if improved:
                    break
            
            if improved:
                base_set = S
                if len(base_set) > best_size:
                    best_size = len(base_set)
                    best_set = base_set[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
    
    # Final verification
    if best_set:
        final_sums = compute_sums(best_set)
        non_empty = final_sums - {0}
        if len(non_empty) != (1 << len(best_set)) - 1:
            # Duplicates found, fallback
            best_set = []
        elif not check_divisibility_fast(non_empty):
            # Divisibility violation, fallback
            best_set = []
    
    # Ensure we return a valid set
    if not best_set or len(best_set) < 1:
        # Fallback: single element is always valid
        return [256]
    
    # Convert to pure Python ints
    return [int(x) for x in best_set]

if __name__ == "__main__":
    res = construct()
    print(res)
