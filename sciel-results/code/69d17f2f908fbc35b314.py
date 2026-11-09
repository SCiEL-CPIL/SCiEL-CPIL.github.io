import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Precompute all valid numbers
    candidates = list(range(2, 129))
    
    # Memoization for (last_value, frozenset_of_products) -> max_additional_length
    # Since products can get large, we might not memoize on the full set if it's too big,
    # but for length ~20-30, the number of distinct states might be manageable if we prune well.
    # Actually, the set of products is the key constraint. 
    # A better state for memoization might be just the 'last value' and the 'set of products'.
    # However, the set of products is large.
    # Let's try a different memoization:
    # State: (last_value, frozenset(products))
    # If we reach the same state, we don't need to re-explore.
    
    memo = {}
    
    def dfs(last_val, prods, current_len):
        # Check time
        if time.time() > deadline:
            return 0
        
        # Pruning: If current_len + (128 - last_val) <= best_global, we can't beat it?
        # No, we can't necessarily add all remaining.
        # Upper bound: current_len + (128 - last_val) is a loose upper bound.
        
        # Check memo
        # Using frozenset of prods as part of key is expensive.
        # Let's see if we can use a hash of the prods set.
        key = (last_val, frozenset(prods))
        
        if key in memo:
            return memo[key]
        
        best_add = 0
        
        # Try adding each candidate > last_val
        for next_val in range(last_val + 1, 129):
            # Check if next_val can be added
            # New products are:
            # 1. next_val
            # 2. next_val * suffix_product for all suffixes of the current sequence ending at last_val
            
            # To compute suffix products efficiently, we would need the sequence, 
            # but we only have the set of products. 
            # Wait, the set of products contains ALL block products.
            # If we add next_val, the new products are:
            # P_new = { next_val * p | p in {1} U {suffix_products_of_seq} }
            # But we don't have the sequence, only the set of all products.
            # This makes it hard to generate the *new* products from just the set.
            
            # Alternative: Pass the sequence or the list of suffix products.
            # Let's change the DFS state to include the sequence or suffix products.
            # State: (last_val, frozenset(all_products), tuple(suffix_products))
            # This is getting complex.
            
            # Let's stick to a simpler DFS that passes the sequence or necessary info.
            # Given the small range (2-128) and target length (~20-30), 
            # passing the list of suffix products is efficient.
            
            # We need to restructure the DFS to pass suffix products.
            # Let's define a helper that takes the current suffix products.
            # Actually, let's just use the sequence in the DFS for simplicity, 
            # as the depth is limited.
            
            # Wait, the current `dfs` signature doesn't have the sequence.
            # Let's rewrite the DFS to include the sequence or suffix products.
            
            pass # Placeholder, will rewrite below
            
    # Refined DFS approach
    # State: (last_val, frozenset(products), tuple(suffix_prods))
    # suffix_prods: products of consecutive blocks ending at the last element.
    # e.g., if seq ends with [a, b], suffix_prods = [b, a*b]
    
    memo = {}
    
    def dfs(last_val, prods, suffix_prods, current_len):
        if time.time() > deadline:
            return 0
            
        key = (last_val, frozenset(prods), suffix_prods)
        if key in memo:
            return memo[key]
            
        best_add = 0
        
        for next_val in range(last_val + 1, 129):
            # Generate new suffix products
            new_suffix_prods = [next_val]
            p = next_val
            for sp in suffix_prods:
                p *= sp
                new_suffix_prods.append(p)
                
            # Check for conflicts
            valid = True
            for np in new_suffix_prods:
                if np in prods:
                    valid = False
                    break
                    
            if valid:
                # Update prods
                new_prods = set(prods)
                new_prods.update(new_suffix_prods)
                
                # Recurse
                res = dfs(next_val, new_prods, tuple(new_suffix_prods), current_len + 1)
                if res + 1 > best_add:
                    best_add = res + 1
        
        memo[key] = best_add
        return best_add

    # Start DFS
    # Initial state: last_val=1, prods=set(), suffix_prods=()
    # We want to find the max length.
    # The DFS returns the max *additional* length from this state.
    # So total length = 1 + dfs(1, set(), (), 0) ? No, the base case returns 0.
    # If we pick a number, we add 1 + dfs(...).
    
    # Let's initialize best_len = 0
    # We can start by trying each first element?
    # Or just call dfs(1, set(), (), 0) which will try all next_vals > 1.
    
    # Note: The memoization key might be large. 
    # frozenset(prods) can be expensive to hash.
    # But the number of states should be pruned heavily.
    
    try:
        max_add = dfs(1, set(), (), 0)
        # The result is the max length of the sequence.
        # We need to reconstruct the sequence?
        # The task asks to RETURN the sequence.
        # The above DFS only returns the length.
        # We need to store the path.
        
        # Let's modify DFS to return the sequence or store it in a global var when improved.
        
    except RecursionError:
        # Fallback if recursion depth is an issue (unlikely for n<128)
        pass

    # Let's implement a version that reconstructs the path.
    memo = {}
    best_seq_found = []
    
    def dfs_path(last_val, prods, suffix_prods, current_seq):
        nonlocal best_seq_found
        
        if time.time() > deadline:
            return 0
            
        key = (last_val, frozenset(prods), suffix_prods)
        if key in memo:
            return memo[key]
            
        best_add = 0
        best_next_seq = None
        
        for next_val in range(last_val + 1, 129):
            new_suffix_prods = [next_val]
            p = next_val
            for sp in suffix_prods:
                p *= sp
                new_suffix_prods.append(p)
                
            valid = True
            for np in new_suffix_prods:
                if np in prods:
                    valid = False
                    break
                    
            if valid:
                new_prods = set(prods)
                new_prods.update(new_suffix_prods)
                
                # Check if this path could potentially beat the current best
                # Upper bound: current_len + 1 + (128 - next_val)
                # If this upper bound <= len(best_seq_found), skip?
                # len(current_seq) + 1 + (128 - next_val) <= len(best_seq_found)
                if len(current_seq) + 1 + (128 - next_val) <= len(best_seq_found):
                    continue
                
                res = dfs_path(next_val, new_prods, tuple(new_suffix_prods), current_seq + [next_val])
                if res + 1 > best_add:
                    best_add = res + 1
                    best_next_seq = current_seq + [next_val]
                    if len(best_next_seq) > len(best_seq_found):
                        best_seq_found = best_next_seq.copy()
        
        memo[key] = best_add
        return best_add

    # Initialize with a greedy solution to have a good lower bound for pruning
    def greedy_construct():
        seq = []
        prods = set()
        last_val = 1
        for x in range(2, 129):
            valid = True
            # Check x
            if x in prods:
                valid = False
            else:
                # Check suffixes
                # We need suffix products of seq
                # For greedy, we can just compute on the fly
                suffix_prod = 1
                for i in range(len(seq) - 1, -1, -1):
                    suffix_prod *= seq[i]
                    if x * suffix_prod in prods:
                        valid = False
                        break
            if valid:
                seq.append(x)
                # Add new prods
                prods.add(x)
                suffix_prod = x
                for i in range(len(seq) - 2, -1, -1):
                    suffix_prod *= seq[i]
                    prods.add(suffix_prod)
                last_val = x
        return seq

    best_seq_found = greedy_construct()
    
    # Run DFS
    # Note: dfs_path modifies best_seq_found
    # We start from last_val=1
    dfs_path(1, set(), (), [])
    
    # Final verification
    def is_valid(seq):
        if not seq:
            return True
        if not all(2 <= x <= 128 for x in seq):
            return False
        if not all(seq[i] < seq[i+1] for i in range(len(seq)-1)):
            return False
        products = set()
        for i in range(len(seq)):
            prod = 1
            for j in range(i, len(seq)):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True

    if not is_valid(best_seq_found):
        # Fallback to a simple valid sequence if something went wrong
        return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
        
    return best_seq_found

if __name__ == "__main__":
    res = construct()
    print(f"Length: {len(res)}")
    print(res)

