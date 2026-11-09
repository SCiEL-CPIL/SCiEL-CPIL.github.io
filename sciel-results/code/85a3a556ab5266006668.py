import math
import time

def construct():
    N = 256
    
    # Bitset representation for the set.
    # We use a Python integer as a bitset. Bit i is set if (i+1) is in the set? 
    # Or bit i corresponds to element i+1? 
    # Let's use bit i for element i+1. So element 1 is bit 0, element 256 is bit 255.
    # This makes checking "is x in set" -> (bits >> (x-1)) & 1.
    
    # Precompute for each x, the list of pairs (y, z) such that {x, y, z} forms an AP.
    # Actually, for adding x, we need to check if there exist y, z in S such that x, y, z form an AP.
    # This is equivalent to checking if for any y in S, z = 2y - x is in S (and z != y, z != x)
    # OR if for any y in S, z = (x+y)/2 is in S (and z != y, z != x)
    # OR if for any y in S, z = 2x - y is in S (and z != y, z != x)
    
    # A faster check using bitsets:
    # Let S be the bitset.
    # To check if adding x creates an AP:
    # 1. Check if there is y in S such that 2y - x is in S.
    #    This is equivalent to checking if (S & (S shifted appropriately)) has any bits set.
    #    Specifically, if we shift S by x, we are checking pairs (y, y+x). 
    #    An AP (y, y+x, y+2x) exists if y, y+x, y+2x are all in S.
    #    This is getting complicated with simple shifts.
    
    # Let's stick to a slightly optimized Python set check but with a better heuristic.
    # Or use the property: 
    # x forms an AP with S iff:
    # - There exists y in S such that 2y - x in S (x is the "end" of AP y, 2y-x, x? No, y, (y+x)/2, x or y, x, 2x-y)
    # Let's just use the standard check but optimize the inner loop.
    
    # Better Idea: Precompute for each x, the set of "conflicts" if x were added.
    # Actually, let's just write a very fast local search.
    
    # 1. Greedy Initialization
    # Sort candidates by 1/x descending (i.e., x ascending)
    S = set()
    bits = 0 # Bitset for S
    
    def is_valid_add(x, current_set, current_bits):
        # Check if adding x to current_set creates a 3-term AP
        # x, y, z form AP if 2y = x + z.
        # So for any y in current_set:
        #   z = 2y - x. If z in current_set and z != x and z != y -> Invalid.
        #   Also, if x is the middle: 2x = y + z -> z = 2x - y. If z in current_set and z != x and z != y -> Invalid.
        #   Also, if x is the start: y, z, x? No, that's covered by 2z = y+x -> z = (y+x)/2.
        
        # Let's iterate through y in current_set.
        for y in current_set:
            # Case 1: y, x, z => z = 2x - y
            z1 = 2 * x - y
            if 1 <= z1 <= N and z1 != x and z1 != y and (current_bits >> (z1-1)) & 1:
                return False
            # Case 2: y, z, x => z = (y+x)/2
            s = y + x
            if s % 2 == 0:
                z2 = s // 2
                if 1 <= z2 <= N and z2 != x and z2 != y and (current_bits >> (z2-1)) & 1:
                    return False
            # Case 3: x, y, z => z = 2y - x
            z3 = 2 * y - x
            if 1 <= z3 <= N and z3 != x and z3 != y and (current_bits >> (z3-1)) & 1:
                return False
        return True

    # Greedy: Add smallest x first
    for x in range(1, N + 1):
        if is_valid_add(x, S, bits):
            S.add(x)
            bits |= (1 << (x - 1))
            
    current_score = sum(1.0 / x for x in S)
    best_score = current_score
    best_set = set(S)
    
    # 2. Local Search
    # We will try to improve the score by swapping elements.
    # A swap (remove x, add y) is beneficial if 1/y > 1/x (i.e., y < x).
    # So we should look for small y not in S that can replace larger x in S.
    
    # To make this fast, we can iterate x in S (largest first) and look for y < x not in S.
    
    # Time limit management
    start_time = time.time()
    timeout = 55.0
    
    # We'll do a few passes of local search.
    # In each pass, we try to find any improving swap.
    
    # To speed up is_valid_add, we can use the bitset directly.
    def is_valid_add_fast(x, bits):
        # Check if adding x to set represented by bits creates AP
        # For each y in set (iterate bits? No, iterate S is faster if S is small, but S can be ~50-100 elements)
        # Let's stick to iterating S.
        pass
        
    # Actually, the previous is_valid_add is O(|S|). |S| is roughly 50-100.
    # 256 * 100 = 25600 checks. Very fast.
    
    # Let's implement a local search that tries all possible swaps in a specific order.
    # Order: Try to replace large x with small y.
    
    # Precompute list of elements not in S
    # This changes as we swap, so we need to be careful.
    
    # Strategy:
    # While time < timeout:
    #   improved = False
    #   For x in sorted(S, reverse=True): # Try to remove large x
    #     For y in range(1, x): # Try to add small y
    #       If y not in S:
    #         If is_valid_add(y, S - {x}):
    #           # Check if swap is valid: Removing x must not break validity? 
    #           # Removing x never breaks validity.
    #           # Adding y to S-{x} must be valid.
    #           # If valid, and 1/y > 1/x (always true since y<x), then swap.
    #           # Perform swap, update score, improved = True, break inner loop.
    #           # If improved, break outer loop and restart scan.
    
    # This is O(|S|^2 * |S|) per pass? No, is_valid_add is O(|S|).
    # So one full pass is O(|S|^3) roughly? 
    # |S| ~ 100. 100^3 = 1,000,000. 
    # 1 million operations is very fast in Python if optimized? Maybe 0.1s.
    # So we can do many passes.
    
    # Let's optimize is_valid_add to use bits for the "in set" check.
    
    def can_add(y, bits, exclude_x=None):
        # Check if y can be added to set represented by bits, 
        # ignoring exclude_x (if we are swapping x out, x is not in the set for the check)
        # Actually, if we are swapping x out, we should check if y is valid in S \ {x}.
        # So we should clear bit x before checking, or pass a modified bits.
        
        # Let's just iterate S and skip x.
        for z in S:
            if z == exclude_x:
                continue
            # Check APs involving y and z
            # 1. z, y, w => w = 2y - z
            w1 = 2 * y - z
            if 1 <= w1 <= N and w1 != y and w1 != z:
                if w1 != exclude_x and ((bits >> (w1-1)) & 1):
                    return False
            # 2. z, w, y => w = (z+y)/2
            s = z + y
            if s % 2 == 0:
                w2 = s // 2
                if 1 <= w2 <= N and w2 != y and w2 != z:
                    if w2 != exclude_x and ((bits >> (w2-1)) & 1):
                        return False
            # 3. y, z, w => w = 2z - y
            w3 = 2 * z - y
            if 1 <= w3 <= N and w3 != y and w3 != z:
                if w3 != exclude_x and ((bits >> (w3-1)) & 1):
                    return False
        return True

    # Local Search Loop
    while time.time() - start_time < timeout:
        improved = False
        # Sort S in descending order to try removing largest elements first
        sorted_S = sorted(S, reverse=True)
        
        for x in sorted_S:
            if improved:
                break
            # Try to replace x with a smaller y
            # y must be < x to improve score
            for y in range(1, x):
                if y in S:
                    continue
                # Check if y can be added to S \ {x}
                if can_add(y, bits, x):
                    # Perform swap
                    S.remove(x)
                    S.add(y)
                    bits &= ~(1 << (x - 1))
                    bits |= (1 << (y - 1))
                    current_score += 1.0/y - 1.0/x
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = set(S)
                    break # Restart search after a swap
        
        if not improved:
            # No more simple swaps improve the score.
            # Try a more complex move? Or break?
            # We can try random swaps to escape local minima?
            # Or just break if no improvement found in a full pass.
            break

    return list(best_set)
