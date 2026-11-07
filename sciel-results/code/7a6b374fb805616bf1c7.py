import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 256
    full_set = set(range(2, N + 1))
    
    def get_conflict_pairs(x, S):
        """
        Returns list of (b, c) in S such that x conflicts with b, c in S.
        Conflict if x|bc, b|xc, or c|xb.
        """
        pairs = []
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            b = S_list[i]
            for j in range(i + 1, n):
                c = S_list[j]
                if (b * c) % x == 0 or (x * c) % b == 0 or (x * b) % c == 0:
                    pairs.append((b, c))
        return pairs

    def can_add(x, S):
        return len(get_conflict_pairs(x, S)) == 0

    best_S = set()
    best_size = 0
    
    for run in range(100):
        if time.time() > deadline:
            break
            
        S = set(full_set)
        
        # 1. Greedy Removal
        while True:
            if time.time() > deadline:
                break
                
            bads = []
            for a in S:
                S_without_a = S - {a}
                # Check if a divides b*c for any b,c in S\{a}
                conflict_found = False
                S_list = list(S_without_a)
                n = len(S_list)
                for i in range(n):
                    b = S_list[i]
                    for j in range(i + 1, n):
                        c = S_list[j]
                        if (b * c) % a == 0:
                            conflict_found = True
                            break
                    if conflict_found:
                        break
                if conflict_found:
                    bads.append(a)
            
            if not bads:
                break
            
            a_to_remove = random.choice(bads)
            S.remove(a_to_remove)
        
        # 2. Local Search
        removed = full_set - S
        
        # Iterate until no change
        # We use a flag to track if we added anything.
        # We also allow swaps to explore.
        
        # To prevent infinite swap loops, we limit the number of consecutive swaps without an addition.
        swap_streak = 0
        max_swap_streak = 50
        
        while time.time() < deadline:
            if swap_streak > max_swap_streak:
                break
                
            rem_list = list(removed)
            random.shuffle(rem_list)
            
            progress = False
            for x in rem_list:
                if time.time() > deadline:
                    break
                
                pairs = get_conflict_pairs(x, S)
                
                if not pairs:
                    # Can add
                    S.add(x)
                    removed.remove(x)
                    progress = True
                    swap_streak = 0
                else:
                    # Try to find a common element in all pairs
                    # Intersection of all pairs
                    # Start with first pair's elements
                    common = set(pairs[0])
                    for b, c in pairs[1:]:
                        common.intersection_update({b, c})
                        if not common:
                            break
                    
                    if common:
                        u = random.choice(list(common))
                        # Verify swap
                        # Check if S \ {u} U {x} is valid
                        # We can just call can_add(x, S - {u})
                        # But we know u hits all pairs, so it should be valid.
                        # However, there might be conflicts NOT involving x?
                        # No, S is valid. Removing u keeps S\{u} valid.
                        # Adding x: conflicts are only with S\{u}.
                        # Since u hit all conflict pairs of x with S, 
                        # and any pair in S\{u} is a pair in S,
                        # if u is in the pair, it's removed.
                        # If u is NOT in the pair, then the pair was not a conflict?
                        # Wait.
                        # If (b,c) is a conflict pair for x in S, and u is in (b,c), then in S\{u}, (b,c) is gone.
                        # Are there new conflict pairs? No, S\{u} is subset of S.
                        # So if u hits ALL conflict pairs in S, then S\{u} has NO conflict pairs with x.
                        # So x can be added.
                        
                        S.remove(u)
                        S.add(x)
                        removed.remove(x)
                        removed.add(u)
                        # This is a swap. Size unchanged.
                        # We don't set progress = True (which implies addition), 
                        # but we do reset swap_streak? 
                        # No, we increment swap_streak.
                        swap_streak += 1
                        # We continue to try other x's?
                        # If we swap, the set S changes.
                        # It might allow other additions.
                        # So we should continue the loop.
                        
            if not progress and swap_streak == 0:
                # No additions and no swaps made in this pass?
                # Actually, if no swaps were made, swap_streak remains 0.
                # If we made swaps, swap_streak > 0.
                # If we made no additions, progress is False.
                # We should break if we made no additions AND no swaps?
                # Or if we made no additions, but made swaps, we might want to continue?
                # But if we keep swapping without adding, we are stuck.
                # So if no additions, we should break after some swaps.
                
                # Let's break if no additions were made in this pass.
                break
            elif not progress:
                # Swaps were made, but no additions.
                # Continue?
                # The condition `if swap_streak > max_swap_streak: break` handles this.
                pass

        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
        if best_size > 60:
            break

    return list(best_S)
