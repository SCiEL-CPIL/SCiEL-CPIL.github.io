import time
import sys

def construct():
    # Set recursion limit just in case, though depth is small
    sys.setrecursionlimit(10000)
    
    # 1. Generate all 2,3,5-smooth numbers <= 128
    smooth = []
    smooth_set = set()
    for a in range(0, 8):
        for b in range(0, 6):
            for c in range(0, 5):
                v = (2 ** a) * (3 ** b) * (5 ** c)
                if 1 <= v <= 128 and v not in smooth_set:
                    smooth_set.add(v)
                    smooth.append(v)
    smooth.sort()
    n_smooth = len(smooth)
    
    # 2. Precompute divisibility conflicts
    # conflict[i][j] is True if smooth[i] divides smooth[j] or vice versa
    conflict = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        si = smooth[i]
        for j in range(i + 1, n_smooth):
            sj = smooth[j]
            if sj % si == 0 or si % sj == 0:
                conflict[i][j] = True
                conflict[j][i] = True
                
    # 3. Find minimal antichain representation for each sum 1..128
    # min_terms[s] = minimum number of terms to sum to s
    # best_rep[s] = tuple of values that achieve this minimum
    min_terms = [100] * 129 # Initialize with a large number
    best_rep = [None] * 129
    
    # We use a recursive search to enumerate all antichains
    # To ensure we find the MINIMUM size, we can just enumerate all valid antichains
    # and keep track of the best (smallest size) for each sum.
    
    # Optimization: 
    # We can prune if current_sum > 128.
    # We can also prune if current_size >= min_terms[current_sum] is not quite right 
    # because a larger size might lead to a different sum, but for a FIXED sum, 
    # we only care about the minimum size.
    # However, we are building the sum incrementally.
    # If we have a partial antichain with sum S and size K, and we know that 
    # min_terms[S] <= K, does it help? No, because adding more numbers changes S.
    # So we must enumerate all antichains with sum <= 128.
    
    # The number of antichains in the divisor lattice of smooth numbers <= 128 is small.
    # Let's estimate: 
    # Size 1: ~70
    # Size 2: ~2000?
    # Size 3: ~10000?
    # It should be fast.
    
    def backtrack(start_idx, current_sum, current_size, current_indices):
        # Update best for current_sum if current_size is smaller
        if current_sum > 0 and current_sum <= 128:
            if current_size < min_terms[current_sum]:
                min_terms[current_sum] = current_size
                best_rep[current_sum] = tuple(smooth[i] for i in current_indices)
        
        # Pruning: if current_sum > 128, stop
        if current_sum > 128:
            return
            
        # Try adding more elements
        # We must pick indices > start_idx to avoid duplicates (combinations, not permutations)
        for i in range(start_idx, n_smooth):
            # Check if adding smooth[i] conflicts with any in current_indices
            valid = True
            for j in current_indices:
                if conflict[i][j]:
                    valid = False
                    break
            
            if not valid:
                continue
                
            new_sum = current_sum + smooth[i]
            if new_sum > 128:
                # Since smooth is sorted, adding larger numbers will only increase sum
                # So we can break here
                break
                
            current_indices.append(i)
            backtrack(i + 1, new_sum, current_size + 1, current_indices)
            current_indices.pop()
            
    # Start with empty set
    # Note: We must handle the case where sum=0 is not a target, so we start recursion 
    # and update inside.
    backtrack(0, 0, 0, [])
    
    # 4. Construct the output
    answers = []
    for n in range(1, 129):
        if best_rep[n] is not None:
            # Return as a list of integers
            answers.append(list(best_rep[n]))
        else:
            # Fallback: This should not happen if the problem is solvable.
            # But for safety, if a representation wasn't found, we try a simple one.
            # However, the problem states "All targets require valid representations".
            # If min_terms[n] is still 100, it means no antichain sums to n.
            # Given the density of smooth numbers, this is highly unlikely for 1..128.
            # But let's add a fallback just in case of a bug.
            # A single term n is valid if n is smooth.
            if n in smooth_set:
                answers.append([n])
            else:
                # Try to find any valid representation, even if not minimal.
                # This is a safety net.
                # We can use a simple greedy or search for this specific n.
                # But since backtrack covered all sums <= 128, this shouldn't happen.
                # We'll just put [1, n-1] if n-1 is smooth and 1 doesn't divide n-1? 
                # 1 divides everything. So [1, x] is never a valid antichain unless x=1 (not distinct).
                # So [1, x] is invalid.
                # We need distinct terms.
                # If we failed, we have a problem. 
                # Let's assume the backtrack worked.
                answers.append([n]) # Placeholder, might be invalid if n not smooth.
                
    return answers

# To verify the logic locally, one might print the total sum
# if __name__ == '__main__':
#     res = construct()
#     total = sum(len(r) for r in res)
#     print(f"Total terms: {total}")
#     for i, r in enumerate(res):
#         # Check validity
#         s = sum(r)
#         if s != i+1:
#             print(f"Error: Sum mismatch for {i+1}: {s} != {i+1}")
#         # Check distinct
#         if len(r) != len(set(r)):
#             print(f"Error: Non-distinct terms for {i+1}")
#         # Check smooth
#         for term in r:
#             t = term
#             while t % 2 == 0: t //= 2
#             while t % 3 == 0: t //= 3
#             while t % 5 == 0: t //= 5
#             if t != 1:
#                 print(f"Error: Non-smooth term {term} for {i+1}")
#         # Check antichain
#         for a in r:
#             for b in r:
#                 if a != b and (a % b == 0 or b % a == 0):
#                     print(f"Error: Divisibility conflict {a}, {b} for {i+1}")
#     print("Validation complete.")
