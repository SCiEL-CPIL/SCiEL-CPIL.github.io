import time
import random

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    def check_violation(element, others):
        """
        Check if 'element' is the arithmetic mean of any subset of 'others'
        with size >= 2.
        Equivalent to: exists subset S of others, |S|>=2, such that sum(S) = |S| * element.
        This is sum(t - element for t in S) = 0.
        
        We use a bitset DP where dp[s] is a bitmask of subset sizes that can achieve sum s.
        Bit k in dp[s] is 1 if sum s is achievable with subset size k.
        """
        if len(others) < 2:
            return False
        
        # Range of sums: t in [1, 512], element in [1, 512].
        # diff = t - element is in [-511, 511].
        # Max size of others is 47 (since element is not in others, and total set size <= 48).
        # Actually others can be up to 47 elements.
        # Sum range: [-47*511, 47*511] approx [-24017, 24017].
        # Let's use a dictionary for sparse representation or a fixed array if dense enough.
        # With 47 elements, the number of reachable sums might be large but bounded.
        # A dict is safer for memory and speed if sparse.
        
        dp = {0: 1}  # sum 0, size 0 (bit 0 set)
        
        for t in others:
            val = t - element
            new_dp = dict(dp)
            for s, mask in dp.items():
                new_s = s + val
                new_mask = mask << 1
                if new_s in new_dp:
                    new_dp[new_s] |= new_mask
                else:
                    new_dp[new_s] = new_mask
            dp = new_dp
            
        # Check if sum 0 is achievable with size >= 2
        if 0 in dp:
            mask = dp[0]
            # We need to check if any bit k >= 2 is set.
            # mask has bit k set if size k is possible.
            # Bit 0 is size 0. Bit 1 is size 1.
            # We want to know if mask has any bit set for k >= 2.
            if mask & ~3:  # ~3 is ...11111100, so it checks bits 2 and above
                return True
        return False

    def check_violation_with_new(element, others, new_elem):
        """
        Check if 'element' is the mean of a subset of (others + [new_elem])
        with size >= 2.
        
        Cases:
        1. Subset does not include new_elem: already checked previously if set was valid.
           But we are adding new_elem, so we must ensure existing elements are still valid.
           Actually, the condition is "No element may be the mean of any subset of OTHER elements".
           When we add new_elem, for an existing element 'a', the set of OTHER elements grows.
           So 'a' might now be the mean of a subset that includes new_elem.
           
        Let S be the subset of OTHER elements for 'a'.
        S is a subset of (current_set \ {a}).
        If new_elem is in S, let S' = S \ {new_elem}. S' is a subset of (current_set \ {a, new_elem}).
        Condition: sum(S) = |S| * a
        sum(S' + [new_elem]) = (|S'| + 1) * a
        sum(S') + new_elem = |S'| * a + a
        sum(S') - |S'| * a = a - new_elem
        
        Let T = S'. We need to check if there exists a subset T of (current_set \ {a, new_elem})
        such that sum(t - a for t in T) = a - new_elem.
        Note: T can be empty. If T is empty, sum=0, so a - new_elem = 0 => a = new_elem, which is false.
        So T must be non-empty?
        Wait, the subset S must have size >= 2.
        If S includes new_elem, then S = {new_elem} U T.
        Size of S is 1 + |T|.
        We need 1 + |T| >= 2 => |T| >= 1.
        So T must be non-empty.
        
        So we check: exists non-empty subset T of (current_set \ {a, new_elem}) such that
        sum(t - a for t in T) = a - new_elem.
        """
        target = element - new_elem
        if target == 0:
            # This would imply element == new_elem, which is impossible.
            return False
            
        dp = {0: 1}  # sum 0, size 0
        for t in others:
            if t == element or t == new_elem:
                continue
            val = t - element
            new_dp = dict(dp)
            for s, mask in dp.items():
                new_s = s + val
                new_mask = mask << 1
                if new_s in new_dp:
                    new_dp[new_s] |= new_mask
                else:
                    new_dp[new_s] = new_mask
            dp = new_dp
            
        if target in dp:
            mask = dp[target]
            # We need size of T >= 1.
            # Bit 0 corresponds to size 0.
            # If mask has any bit set for k >= 1, it's a violation.
            if mask & ~1:
                return True
        return False

    def try_greedy(order, max_size=48):
        selected = []
        for x in order:
            if len(selected) >= max_size:
                break
                
            # 1. Check if x is the mean of a subset of selected (size >= 2)
            if check_violation(x, selected):
                continue
                
            # 2. Check if any existing element a in selected is now the mean of a subset including x
            valid = True
            for a in selected:
                if check_violation_with_new(a, selected, x):
                    valid = False
                    break
                    
            if valid:
                selected.append(x)
                
        return selected

    best_set = []
    
    # Strategy 1: Randomized greedy
    # We try multiple random permutations.
    # To speed up, we can also try biased orders (e.g., powers of 2, or spread out).
    
    # A good initial set is powers of 2, but they don't maximize density.
    # Let's just run many random restarts.
    
    num_restarts = 500
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        # Occasionally try a structured order?
        # e.g., sort by value mod something?
        # For now, pure random is robust.
        
        current_set = try_greedy(candidates)
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
            # If we reached the max, we can stop early?
            # The task says "Maximize cardinality". Max is 48.
            # If we hit 48, we are done.
            if len(best_set) == 48:
                break

    # Strategy 2: Local Search / Hill Climbing
    # If we didn't reach 48, try to improve the best set found.
    # Try swapping elements or adding/removing.
    
    # Actually, a better approach for local search:
    # Start with best_set.
    # Try to add an element from 1..512 not in best_set.
    # If adding fails, try swapping an element in best_set with one from outside.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding
        for x in range(1, 513):
            if x in best_set:
                continue
            if len(best_set) >= 48:
                break
                
            # Check if x can be added
            if check_violation(x, best_set):
                continue
                
            valid = True
            for a in best_set:
                if check_violation_with_new(a, best_set, x):
                    valid = False
                    break
                    
            if valid:
                best_set.append(x)
                improved = True
                break  # Restart the loop to try adding more
                
        if len(best_set) == 48:
            break
            
        # Try swapping
        if not improved:
            # Try to swap an element y in best_set with x not in best_set
            # This is expensive: O(N * N * check_cost)
            # N=48, candidates=512. 48*512 = 24576 swaps.
            # Each swap requires checking validity of the new set.
            # We can optimize by only checking the changed parts?
            # If we remove y and add x:
            # 1. x must not be mean of (best_set - {y})
            # 2. For each a in (best_set - {y}), a must not be mean of ((best_set - {y}) - {a}) U {x}
            #    This includes the case where x is in the subset.
            
            # Let's implement a helper to check if a set is valid.
            # But that's O(N^2 * DP).
            # Maybe just try a few random swaps?
            
            # Let's try random swaps from the candidate pool.
            candidates = [x for x in range(1, 513) if x not in best_set]
            random.shuffle(candidates)
            
            swap_found = False
            for x in candidates[:50]:  # Limit swaps to try
                if time.time() > deadline:
                    break
                for y in best_set:
                    new_set = [z for z in best_set if z != y] + [x]
                    
                    # Check validity of new_set
                    # This is a full check.
                    # To speed up, we can check:
                    # 1. Is x mean of any subset of new_set - {x}?
                    # 2. Is any a in new_set - {x} mean of any subset of new_set - {a}?
                    
                    # This is expensive. Let's just do it for a few swaps.
                    valid = True
                    for a in new_set:
                        others = [z for z in new_set if z != a]
                        if check_violation(a, others):
                            valid = False
                            break
                            
                    if valid:
                        best_set = new_set
                        improved = True
                        swap_found = True
                        break
                if swap_found:
                    break

    return best_set

if __name__ == "__main__":
    # Quick self-test
    s = construct()
    print(f"Found set of size {len(s)}")
    # Verify
    def verify(s):
        for a in s:
            others = [x for x in s if x != a]
            if check_violation(a, others):
                return False
        return True
    print("Valid:", verify(s))
