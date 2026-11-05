"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    best_sequence = []
    best_length = 0
    
    # Global state for DFS
    current_seq = []
    existing_products = set()
    
    # Candidate numbers are 2 to 128
    # We need strictly increasing sequence.
    
    def dfs(last_val, remaining_candidates):
        nonlocal best_sequence, best_length, deadline
        
        # Check time
        if time.time() > deadline:
            return False # Signal timeout
        
        # Update best if current is better
        if len(current_seq) > best_length:
            best_length = len(current_seq)
            best_sequence = current_seq[:]
        
        # If no candidates left, or we've hit max possible length (theoretical max is 127, but practically much lower)
        if not remaining_candidates:
            return True
            
        # Optimistic bound: even if we pick all remaining, can we beat best?
        if len(current_seq) + len(remaining_candidates) <= best_length:
            return True
            
        # Try to add the next candidate
        # We can choose to skip the first candidate and try the next one,
        # or add the first candidate and recurse.
        # This is a standard "subset" style DFS but with order constraint.
        # Since the sequence must be strictly increasing, and we process candidates in increasing order,
        # if we pick a candidate, the next must be larger.
        
        # Option 1: Skip the first candidate
        # We don't need to explicitly "skip" in a way that changes state, 
        # we just recurse on the rest of the list.
        # But we must ensure we explore the possibility of including the first candidate too.
        
        # Let's iterate through candidates to pick the next one.
        # This is O(N) at each level, which might be slow.
        # Better: Binary choice? No, because we can pick ANY next element from the remaining set.
        # However, since the sequence must be increasing, if we pick $x_i$, we can only pick from $x_{i+1} \dots x_{end}$.
        
        # So, for each candidate $x$ in remaining_candidates:
        # 1. Check if $x$ can be appended.
        # 2. If yes, append, recurse on remaining_candidates after $x$.
        # 3. Backtrack.
        
        # To optimize, we can break early if adding $x$ fails and $x$ is large? No.
        # We should probably try smaller $x$ first as they leave more options?
        # Or larger $x$ to fill the range?
        # Usually, for "maximize length", picking smaller numbers is better because they allow more numbers to be picked later?
        # Actually, picking smaller numbers creates smaller products. Large numbers create large products.
        # Collisions are between products.
        # Let's just iterate.
        
        for i in range(len(remaining_candidates)):
            x = remaining_candidates[i]
            
            # Check if we can append x
            # New products: x, x * last, x * last * second_last, ...
            # We need to check if any of these are in existing_products.
            
            # Calculate new products
            new_products = []
            prod = x
            # Multiply by elements in current_seq in reverse order
            for a in reversed(current_seq):
                new_products.append(prod)
                prod *= a
            new_products.append(prod)
            
            # Check collision
            # Optimization: If any new_product is in existing_products, skip
            # Also, new_products must be distinct among themselves?
            # Since x > 0 and sequence is strictly increasing, 
            # x < x*last < x*last*second_last ... 
            # So new_products are strictly increasing. No internal collisions.
            
            if any(p in existing_products for p in new_products):
                continue
                
            # Valid move
            current_seq.append(x)
            existing_products.update(new_products)
            
            # Recurse on the rest of the candidates (those greater than x)
            # remaining_candidates[i+1:]
            # We pass a slice to avoid mutation issues, though slicing is O(N).
            # Given N <= 127, this is acceptable.
            dfs(x, remaining_candidates[i+1:])
            
            # Backtrack
            current_seq.pop()
            existing_products.difference_update(new_products)
            
            # Check time again after recursion
            if time.time() > deadline:
                return False
                
        return True

    # Initial candidates: 2 to 128
    initial_candidates = list(range(2, 129))
    
    # Start DFS
    # We don't need to pick the first element in the loop if we start with empty seq.
    # The loop in dfs handles picking the first element from remaining_candidates.
    dfs(0, initial_candidates)
    
    return best_sequence
# EVOLVE-BLOCK-END
