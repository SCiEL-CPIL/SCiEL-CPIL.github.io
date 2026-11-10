import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave 5s margin for safety
    
    n = 256
    candidates = list(range(1, n + 1))
    
    # Bitmask helpers
    # Max sum: 256 + 256 = 512. We need bits 0..511.
    # Max diff: 255. We need bits 0..255.
    
    def can_add_bitmask(x, S, sums_mask, diffs_mask, other_diffs_mask):
        # S is a list of elements currently in the set
        # Check Sidon property (sums)
        # New sums are x+a for a in S, and 2x
        # We need to check if any (x+a) is in sums_mask
        # And if any (x+a) == (x+b) -> a==b (ok)
        # And if (x+a) == 2x -> a==x (x not in S, ok)
        
        # Efficient check:
        # We can't just shift the mask easily because we need to check specific values x+a.
        # However, iterating S is O(|S|). |S| <= 64.
        # Bitmask check is O(1) per element.
        
        for a in S:
            s = x + a
            if (sums_mask >> s) & 1:
                return False
            d = abs(x - a)
            if (other_diffs_mask >> d) & 1:
                return False
        
        # Check 2x
        s2 = 2 * x
        if (sums_mask >> s2) & 1:
            return False
            
        return True

    def get_sums_mask(S):
        mask = 0
        for i in range(len(S)):
            for j in range(i, len(S)):
                mask |= (1 << (S[i] + S[j]))
        return mask

    def get_diffs_mask(S):
        mask = 0
        for i in range(len(S)):
            for j in range(len(S)):
                if i != j:
                    d = abs(S[i] - S[j])
                    mask |= (1 << d)
        return mask

    def greedy_construct(candidates_order):
        A = []
        B = []
        sums_A = 0
        sums_B = 0
        diffs_A = 0
        diffs_B = 0
        used = set()
        
        for x in candidates_order:
            # Try A
            if can_add_bitmask(x, A, sums_A, diffs_A, diffs_B):
                # Add to A
                A.append(x)
                # Update sums_A
                for a in A[:-1]: # a is x, so we iterate previous elements
                    sums_A |= (1 << (x + a))
                sums_A |= (1 << (2 * x))
                # Update diffs_A
                for a in A[:-1]:
                    d = abs(x - a)
                    diffs_A |= (1 << d)
                used.add(x)
                continue
            
            # Try B
            if can_add_bitmask(x, B, sums_B, diffs_B, diffs_A):
                # Add to B
                B.append(x)
                # Update sums_B
                for b in B[:-1]:
                    sums_B |= (1 << (x + b))
                sums_B |= (1 << (2 * x))
                # Update diffs_B
                for b in B[:-1]:
                    d = abs(x - b)
                    diffs_B |= (1 << d)
                used.add(x)
                continue
        
        return A, B, used

    def local_search(A, B, used, sums_A, sums_B, diffs_A, diffs_B, time_limit):
        # Try to improve by swapping elements
        # We maintain the masks to allow fast checks
        # Note: sums_A, sums_B, diffs_A, diffs_B are current masks
        
        best_size = len(A) + len(B)
        best_A = A[:]
        best_B = B[:]
        best_sums_A = sums_A
        best_sums_B = sums_B
        best_diffs_A = diffs_A
        best_diffs_B = diffs_B
        best_used = set(used)
        
        # Pre-compute unused list
        all_nums = set(range(1, 257))
        
        while time.time() < time_limit:
            unused_list = list(all_nums - used)
            if not unused_list:
                break
            
            # Pick a random element from A or B to potentially swap/remove
            if A and B:
                # 50% chance swap in A, 50% in B
                if random.random() < 0.5:
                    set_idx = 0
                    S = A
                    sums_S = sums_A
                    diffs_S = diffs_A
                    other_diffs = diffs_B
                    other_sums = sums_B
                else:
                    set_idx = 1
                    S = B
                    sums_S = sums_B
                    diffs_S = diffs_B
                    other_diffs = diffs_A
                    other_sums = sums_A
            elif A:
                set_idx = 0
                S = A
                sums_S = sums_A
                diffs_S = diffs_A
                other_diffs = diffs_B
                other_sums = sums_B
            elif B:
                set_idx = 1
                S = B
                sums_S = sums_B
                diffs_S = diffs_B
                other_diffs = diffs_A
                other_sums = sums_A
            else:
                break
            
            if not S:
                continue

            # Pick random element to remove
            remove_idx = random.randint(0, len(S) - 1)
            elem_to_remove = S[remove_idx]
            
            # Temporarily remove
            # This is complex with masks. 
            # Strategy: Reconstruct masks from S after modification?
            # |S| is small (<64). Reconstructing masks is O(|S|^2) ~ 4000 ops.
            # This is fast enough.
            
            # 1. Remove elem_to_remove from S
            S.pop(remove_idx)
            used.discard(elem_to_remove)
            
            # 2. Try to add a random unused element to S
            # Or try to add ANY unused element that fits
            # To maximize chance of improvement, try a few random unused elements
            
            found_improvement = False
            # Try up to 10 random unused elements
            for _ in range(10):
                if not unused_list:
                    break
                x = random.choice(unused_list)
                
                # Check if x can be added to S (current state)
                if can_add_bitmask(x, S, sums_S, diffs_S, other_diffs):
                    # Add x to S
                    S.append(x)
                    used.add(x)
                    
                    # Update masks
                    # We need to update sums_S and diffs_S
                    # Since we removed an element, sums_S is now invalid (still has old sums).
                    # It's safer to recompute masks from S.
                    sums_S = get_sums_mask(S)
                    diffs_S = get_diffs_mask(S)
                    
                    # Check if we can add MORE elements now?
                    # This is the "shake" part.
                    # Try to greedily add remaining unused elements
                    # We need to be careful not to spend too much time here.
                    
                    # Let's just accept this swap for now and see if size increased?
                    # No, size is same. 
                    # The benefit is that the new configuration might allow adding OTHER elements.
                    
                    # Let's try a quick greedy pass on the remaining unused
                    # This might be expensive.
                    
                    found_improvement = True
                    break
            
            if found_improvement:
                # Update the "current" state variables
                if set_idx == 0:
                    A = S
                    sums_A = sums_S
                    diffs_A = diffs_S
                else:
                    B = S
                    sums_B = sums_S
                    diffs_B = diffs_S
                
                # Now, try to add any unused elements greedily
                # This is where size can increase
                # We need to re-fetch unused list
                unused_list = list(all_nums - used)
                for x in unused_list:
                    if can_add_bitmask(x, A, sums_A, diffs_A, diffs_B):
                        A.append(x)
                        sums_A |= (1 << (2*x))
                        for a in A[:-1]:
                            sums_A |= (1 << (x+a))
                            diffs_A |= (1 << abs(x-a))
                        used.add(x)
                    elif can_add_bitmask(x, B, sums_B, diffs_B, diffs_A):
                        B.append(x)
                        sums_B |= (1 << (2*x))
                        for b in B[:-1]:
                            sums_B |= (1 << (x+b))
                            diffs_B |= (1 << abs(x-b))
                        used.add(x)
                
                current_size = len(A) + len(B)
                if current_size > best_size:
                    best_size = current_size
                    best_A = A[:]
                    best_B = B[:]
                    best_sums_A = sums_A
                    best_sums_B = sums_B
                    best_diffs_A = diffs_A
                    best_diffs_B = diffs_B
                    best_used = set(used)
            else:
                # Revert removal
                S.insert(remove_idx, elem_to_remove)
                used.add(elem_to_remove)
                # Masks are still valid as they were before removal? 
                # No, we didn't change masks during the failed attempts.
                # So sums_S, diffs_S are still correct for the original S.
                pass

        return best_A, best_B, best_size

    best_size = 0
    best_A = []
    best_B = []
    
    # Phase 1: Greedy Restarts
    # Do a few greedy restarts to find a good starting point
    num_greedy_restarts = 20
    for i in range(num_greedy_restarts):
        if time.time() > time_limit - 10: # Keep 10s for local search
            break
        random.shuffle(candidates)
        A, B, used = greedy_construct(candidates)
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A
            best_B = B
            # We don't store masks here, local_search will recompute or we pass lists
            # Let's recompute masks for the best found so far for the local search phase
            # Actually, local_search takes A, B and computes masks?
            # My local_search signature above assumes masks are passed.
            # Let's adjust: compute masks for best_A, best_B before calling local_search
    
    if best_size == 0:
        return [[1], [2]]
    
    # Recompute masks for the best solution found in Phase 1
    sums_A = get_sums_mask(best_A)
    sums_B = get_sums_mask(best_B)
    diffs_A = get_diffs_mask(best_A)
    diffs_B = get_diffs_mask(best_B)
    used = set(best_A) | set(best_B)
    
    # Phase 2: Local Search
    # Run local search on the best solution found
    final_A, final_B, final_size = local_search(
        best_A, best_B, used, sums_A, sums_B, diffs_A, diffs_B, time_limit
    )
    
    if final_size > best_size:
        return [final_A, final_B]
    else:
        return [best_A, best_B]

# Fallback if construct fails or returns empty
if __name__ == "__main__":
    res = construct()
    print(res)
