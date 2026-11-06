import time
import random
import sys

def _is_valid(seq):
    """Check if the sequence has unique block products."""
    n = len(seq)
    products = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True

def _greedy_construct():
    """Greedy baseline."""
    values = []
    for x in range(2, 129):
        candidate = values + [x]
        if _is_valid(candidate):
            values = candidate
    return values

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s kill
    
    # 1. Greedy Initialization (fast, provides a valid baseline)
    best_seq = _greedy_construct()
    best_len = len(best_seq)
    
    # 2. Beam Search to find a better configuration
    # We use the insight that maintaining suffix products and a set of all products
    # allows for O(L) or O(L^2) validity checks during expansion, which is faster
    # than re-validating the whole sequence from scratch in a local search loop
    # when exploring many branches.
    
    # State: (seq_tuple, prod_set, suf_prods_tuple)
    # suf_prods: products of suffixes of the sequence.
    # If seq = [a, b, c], suf_prods = [c, b*c, a*b*c]
    # When adding x:
    # new_sufs = [x] + [x * p for p in suf_prods]
    # new_prods = prod_set.union(new_sufs)
    # Collision if prod_set.intersection(new_sufs) is not empty.
    
    # We will run beam search from scratch, but we can initialize the beam with
    # the greedy result if we want, or just start fresh.
    # Starting fresh is safer to ensure we don't get stuck in a local optimum of the greedy.
    # However, the greedy result is a good lower bound.
    
    # Let's use a beam search that iterates x from 2 to 128.
    # To speed up, we use the vector representation of products?
    # No, the integer products can get huge.
    # But for the "suffix" approach, we just need to check if the new suffixes collide with existing products.
    # The existing products are stored in a set.
    # The new suffixes are just the new elements multiplied by the old suffixes.
    
    # This is effectively what Inspiration 1 did.
    # Let's implement that, but with a tighter time budget and maybe a wider beam.
    
    # To combine with the current program's "local search" idea:
    # The beam search explores the space of subsets.
    # If the beam search finishes and finds a sequence, we can then try to improve it further
    # using the local search (insertions/swaps) from the current program, but that might be redundant
    # if the beam search is already exploring widely.
    
    # Actually, the beam search is a constructive heuristic. It builds the sequence element by element.
    # The local search in the current program tries to modify an existing sequence.
    # The beam search is likely more powerful for finding the global optimum because it doesn't get
    # stuck in a local optimum of "insertions" as easily, provided the beam width is sufficient.
    
    # Let's run the beam search.
    
    K = 1000  # Beam width. 500 was used in inspiration 1. Let's try 1000.
    
    # Initial state
    # seq: tuple
    # prods: set of products
    # sufs: tuple of suffix products
    
    beam = [ ((), set(), ()) ]
    
    for x in range(2, 129):
        if time.time() > deadline:
            break
            
        new_beam = []
        
        for seq, prods, sufs in beam:
            # Option 1: Exclude x
            # Prune if we can't beat best_len
            # Max possible length: len(seq) + (128 - x)
            if len(seq) + (128 - x) >= best_len:
                new_beam.append((seq, prods, sufs))
            
            # Option 2: Include x
            # Calculate new suffixes
            # new_sufs = (x,) + tuple(x * p for p in sufs)
            # Check collision
            # If any element in new_sufs is in prods, it's a collision.
            
            # Optimization: 
            # new_sufs[0] is x.
            # new_sufs[i] is x * sufs[i-1].
            
            # We can check collision without building the full tuple if we want, 
            # but building the tuple is O(L).
            
            new_sufs_list = [x]
            for p in sufs:
                new_sufs_list.append(x * p)
            
            # Check if any new_sufs_list element is in prods
            # Since prods is a set, this is O(L) lookups.
            # But we can also check if the set of new_sufs_list intersects with prods.
            # set(new_sufs_list).intersection(prods)
            # This creates a new set, which is O(L).
            
            # Is it possible that new_sufs_list has duplicates?
            # sufs are distinct because seq is strictly increasing and elements > 1.
            # x > 1. So x * p are distinct.
            # x is distinct from x * p because p > 1 (since p is a product of at least one element >= 2).
            # Wait, if sufs is empty, new_sufs_list is [x].
            # If sufs is not empty, p >= 2, so x*p >= 2x > x.
            # So new_sufs_list elements are distinct.
            
            if not prods.intersection(new_sufs_list):
                new_prods = prods.union(new_sufs_list)
                new_sufs = tuple(new_sufs_list)
                new_seq = seq + (x,)
                
                if len(new_seq) > best_len:
                    best_len = len(new_seq)
                    best_seq = list(new_seq)
                
                new_beam.append((new_seq, new_prods, new_sufs))
        
        if not new_beam:
            break
            
        # Prune new_beam
        # Sort by length descending, then by last element ascending (to keep smaller values for future growth)
        def sort_key(state):
            seq = state[0]
            last = seq[-1] if seq else 0
            return (-len(seq), last)
            
        new_beam.sort(key=sort_key)
        beam = new_beam[:K]
        
        # Early stopping
        if beam:
            max_current_len = max(len(s[0]) for s in beam)
            if max_current_len + (128 - x) < best_len:
                break

    # 3. Local Search Refinement
    # The beam search might leave some numbers out that could be added if we rearrange slightly?
    # No, the sequence must be strictly increasing. The order is fixed by the values.
    # So the "sequence" is just a subset of [2, 128].
    # The beam search explores subsets.
    # If the beam search is saturated, local search (insertions) is equivalent to what the beam search
    # would have done if it had a wider beam or more time.
    # However, the beam search might have dropped a promising state due to beam width.
    # A local search starting from the best found state might be able to "fix" a bad choice
    # by swapping elements?
    # Swapping elements changes the sequence order? No, the sequence is sorted.
    # So swapping values in the sorted list is equivalent to replacing one element with another.
    # This is what the current program's _try_swap does.
    
    # Let's try a limited local search to see if we can squeeze out a longer sequence.
    # We'll use the best_seq from beam search.
    
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            candidates = [x for x in range(2, 129) if x not in best_seq]
            random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline:
                    break
                # Try inserting x
                # Since best_seq is sorted, we can find the insertion point.
                # But we can just create a new list and sort it?
                # No, best_seq is sorted.
                # Let's use a helper to try insertion.
                
                # Simple insertion check
                # Find where x would go
                lo, hi = 0, len(best_seq)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if best_seq[mid] < x:
                        lo = mid + 1
                    else:
                        hi = mid
                insert_idx = lo
                
                new_seq = best_seq[:insert_idx] + [x] + best_seq[insert_idx:]
                if _is_valid(new_seq):
                    best_seq = new_seq
                    best_len = len(best_seq)
                    improved = True
                    break # Restart loop
            
            if not improved:
                # Try swaps?
                # A swap replaces one element with another.
                # This doesn't increase length, but might enable a future insertion.
                # Given the time constraints, and that beam search already explored many configurations,
                # swaps are less likely to yield a jump.
                # But let's try a few random swaps if we have time.
                
                if time.time() < deadline - 5:
                    swap_found = False
                    for i in range(len(best_seq)):
                        if time.time() > deadline:
                            break
                        # Try swapping best_seq[i] with a random candidate
                        # We need to maintain sorted order.
                        # So the new value must be > best_seq[i-1] and < best_seq[i+1].
                        
                        # Find valid range for swap
                        low_bound = best_seq[i-1] if i > 0 else 1
                        high_bound = best_seq[i+1] if i < len(best_seq)-1 else 129
                        
                        # Candidates in (low_bound, high_bound)
                        valid_candidates = [x for x in candidates if low_bound < x < high_bound]
                        if not valid_candidates:
                            continue
                            
                        random.shuffle(valid_candidates)
                        for x in valid_candidates[:5]: # Limit to 5
                            new_seq = best_seq[:i] + [x] + best_seq[i+1:]
                            if _is_valid(new_seq):
                                best_seq = new_seq
                                swap_found = True
                                improved = True
                                break
                        if improved:
                            break
                    # If we swapped, we should try insertions again.
                    # The loop above will handle that if we set improved=True.
                    # But we broke out of the insertion loop.
                    # The outer while loop will restart.
                    
                    if not improved:
                         break

    return best_seq
