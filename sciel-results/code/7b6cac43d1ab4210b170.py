import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    # Bitset for reachable sums
    # We want to know if a sum is reachable.
    # Also want to find a specific conflict.
    
    def get_conflict(S, target=200):
        """
        Returns a set of elements from S that sum to target, or None if no such subset exists.
        Uses DP to find min cardinality conflict.
        """
        if not S:
            return None
            
        INF = float('inf')
        dp = [INF] * (target + 1)
        parent = [None] * (target + 1)
        dp[0] = 0
        
        elements = sorted(S)
        for x in elements:
            if x > target:
                continue
            # Iterate backwards
            for s in range(target - x, -1, -1):
                if dp[s] != INF:
                    new_s = s + x
                    if new_s <= target and dp[s] + 1 < dp[new_s]:
                        dp[new_s] = dp[s] + 1
                        parent[new_s] = (s, x)
        
        if dp[target] == INF:
            return None
        
        # Reconstruct
        conflict = set()
        s = target
        while s > 0:
            if parent[s] is None:
                return None
            prev_s, x = parent[s]
            conflict.add(x)
            s = prev_s
        return conflict

    def is_valid(S):
        return get_conflict(S) is None

    # 1. Construct a good baseline
    # Based on analysis: {1, 2, 3, 4} U {66, ..., 98} is valid and size 37.
    # Let's try to construct a slightly better one or just use this.
    # {1, 2, 3, 4, 66, ..., 98}
    baseline = set([1, 2, 3, 4])
    for i in range(66, 99): # 66 to 98
        baseline.add(i)
        
    # Verify baseline
    if not is_valid(baseline):
        # Fallback to simple greedy if construction fails (should not)
        baseline = set()
        reachable = {0}
        for x in range(100, 0, -1):
            new = {s + x for s in reachable if s + x <= 200}
            if 200 not in new:
                baseline.add(x)
                reachable |= new

    best_set = list(baseline)
    
    # 2. Search: Try to add elements from the complement
    # Candidates are elements not in baseline
    candidates = [x for x in range(1, 101) if x not in baseline]
    
    # Sort candidates to try "promising" ones first?
    # Smaller numbers might be easier to fit?
    candidates.sort()
    
    def search_add(current_set, candidates_list, depth):
        nonlocal best_set
        
        if time.time() > deadline:
            return
        
        if len(current_set) > len(best_set):
            best_set = list(current_set)
            
        # Try to add each candidate
        for i, c in enumerate(candidates_list):
            if time.time() > deadline:
                return
                
            new_set = current_set | {c}
            conflict = get_conflict(new_set)
            
            if conflict is None:
                # Valid, recurse
                search_add(new_set, candidates_list[i+1:], depth + 1)
            else:
                # Conflict found.
                # To include c, we must remove at least one element from conflict \ {c}
                # We branch on removing each x in conflict \ {c}
                # But we must ensure we don't remove c itself (we want to add it)
                # Actually, if we remove x, the set becomes new_set - {x}
                # This set might still have conflicts involving other elements?
                # get_conflict finds ONE conflict. Removing x might break it, but others might remain.
                # However, if we remove x, we can just call search_add again on the resulting set?
                # No, search_add assumes the set is VALID.
                # new_set - {x} might be invalid.
                
                # Better:
                # If new_set is invalid, we cannot add c directly.
                # We can try to add c AND remove x.
                # Let candidate_new_set = new_set - {x}.
                # Check if candidate_new_set is valid.
                # If valid, we can recurse.
                # If invalid, we have a new conflict.
                
                # To keep it simple and within time:
                # Try removing each x in conflict (excluding c)
                # and check if the result is valid.
                # If valid, recurse.
                
                removals = list(conflict)
                if c in removals:
                    removals.remove(c)
                    
                for x in removals:
                    if time.time() > deadline:
                        return
                    repaired_set = new_set - {x}
                    # Check validity of repaired_set
                    # This is expensive if done naively.
                    # But repaired_set is small change.
                    # Just call is_valid?
                    if is_valid(repaired_set):
                         # Recurse with repaired_set
                         # Note: We removed x, so x is no longer in the set.
                         # But x might be a candidate we want to add later?
                         # Our search structure is "add from candidates".
                         # If we remove x from the set, x is effectively "rejected" for this branch?
                         # Or can we add it back?
                         # If we add it back, we are back to a conflict.
                         # So in this branch, x is excluded.
                         
                         # We need to update candidates_list to exclude x?
                         # Or just pass the set and let the recursion handle it?
                         # The recursion adds from candidates_list.
                         # If x is in candidates_list, it might be added again.
                         # We should remove x from candidates for this branch.
                         
                         new_candidates = [cc for cc in candidates_list[i+1:] if cc != x]
                         search_add(repaired_set, new_candidates, depth + 1)
                         
                # Also option: Don't add c at all
                # This is handled by the loop moving to next i
                # But we must also explore branches where we DON'T add c.
                # The loop `for i, c in enumerate...` naturally skips c if we don't recurse on it.
                # However, the recursion `search_add(new_set, candidates_list[i+1:])` assumes c is added.
                # If we don't add c, we just continue the loop.
                # But we also need to explore the branch where we add c and remove x.
                # That is done in the `for x in removals` loop.

    # Start search
    # We start with baseline.
    # Candidates are sorted.
    search_add(baseline, candidates, 0)
    
    return best_set

# EVOLVE-BLOCK-END
