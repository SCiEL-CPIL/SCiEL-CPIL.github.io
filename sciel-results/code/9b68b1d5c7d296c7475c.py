import time
import random

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin before 60s/90s limit
    
    def is_valid(sequence):
        """
        Checks if the sequence has distinct products for all non-empty consecutive blocks.
        Returns True if valid, False otherwise.
        """
        n = len(sequence)
        if n == 0:
            return True
        products = set()
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= sequence[j]
                if p in products:
                    return False
                products.add(p)
        return True

    def greedy_construct():
        """
        Builds a sequence greedily.
        """
        values = []
        products = set()
        for x in range(2, 129):
            valid = True
            # Check new products formed by appending x
            # New products are: x, s[-1]*x, s[-2]*s[-1]*x, ...
            p = x
            # We need to check these against existing products
            # And also, do these new products collide with each other?
            # No, because they are multiples of x by different suffixes, and suffixes are distinct.
            # Wait, s[-1]*x and s[-2]*s[-1]*x are distinct.
            # So we just check against `products`.
            
            # However, we also need to ensure that the *new* products don't collide with *each other*?
            # They are $x, s_{L-1}x, s_{L-2}s_{L-1}x, \dots$
            # Since $s_i > 1$, these are strictly increasing. So they are distinct from each other.
            # So we only need to check against `products`.
            
            # But wait, the definition of `products` in the greedy loop above only updated with tails.
            # Let's re-verify the greedy logic.
            # If we append x, the new block products are:
            # 1. [x] -> product x
            # 2. [s_last, x] -> product s_last * x
            # ...
            # k. [s_{L-k+1}, ..., s_last, x]
            
            # We need to ensure none of these are in the set of *all* previous block products.
            
            p = x
            if p in products:
                valid = False
            else:
                for a in reversed(values):
                    p *= a
                    if p in products:
                        valid = False
                        break
            
            if valid:
                values.append(x)
                # Update products set
                # We need to add the new products to the set
                # The new products are exactly the `p` values we computed?
                # No, we broke early if invalid. If valid, we need to add all of them.
                # Let's re-calculate or store them.
                new_prods = []
                p = x
                new_prods.append(p)
                for a in reversed(values[:-1]): # values[:-1] because we haven't appended x yet? 
                    # Wait, I appended x in the `if valid` block? No, I check first.
                    # Let's refactor.
                    pass
                
                # Refactored Greedy:
                # 1. Calculate new products list.
                # 2. Check if any is in `products`.
                # 3. If not, add x to values, add new products to `products`.
                
                # Let's just use a helper to get new products.
                pass
            
        # The above logic was a bit messy. Let's write a clean greedy.
        values = []
        products = set()
        for x in range(2, 129):
            new_prods = []
            p = x
            new_prods.append(p)
            for a in reversed(values):
                p *= a
                new_prods.append(p)
            
            if not any(p in products for p in new_prods):
                values.append(x)
                products.update(new_prods)
        
        return values

    # Initial sequence
    seq = greedy_construct()
    
    # Local Search
    # We will try to improve the sequence.
    # Move 1: Insert a new element (1-in)
    # Move 2: Replace an element (1-out/1-in)
    # Move 3: Replace two consecutive elements (2-out/2-in)
    # Move 4: Remove an element (1-out) - usually bad for length, but might enable 2-in later?
    #          Actually, if we are stuck, we might need to remove something to allow two insertions.
    #          So 1-out/2-in is the key length-increasing move.
    
    def get_unused(seq):
        used = set(seq)
        return [x for x in range(2, 129) if x not in used]

    def try_insert(seq):
        """Try to insert any unused number at any position."""
        unused = get_unused(seq)
        n = len(seq)
        # Randomize order of positions and candidates to escape local minima
        random.shuffle(unused)
        positions = list(range(n + 1))
        random.shuffle(positions)
        
        for pos in positions:
            for x in unused:
                # Check bounds
                if pos > 0 and seq[pos-1] >= x:
                    continue
                if pos < n and seq[pos] <= x:
                    continue
                
                new_seq = seq[:pos] + [x] + seq[pos:]
                if is_valid(new_seq):
                    return new_seq
        return None

    def try_replace(seq):
        """Try to replace an element with a different unused number."""
        unused = get_unused(seq)
        n = len(seq)
        if n == 0:
            return None
        
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            left_bound = seq[i-1] if i > 0 else 1
            right_bound = seq[i+1] if i < n-1 else 129
            
            candidates = [x for x in unused if left_bound < x < right_bound]
            random.shuffle(candidates)
            
            for x in candidates:
                if x == seq[i]:
                    continue
                new_seq = seq[:i] + [x] + seq[i+1:]
                if is_valid(new_seq):
                    return new_seq
        return None

    def try_1out_2in(seq):
        """Remove one element, insert two new ones."""
        n = len(seq)
        if n == 0:
            return None
        
        unused = get_unused(seq)
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            left_bound = seq[i-1] if i > 0 else 1
            right_bound = seq[i+1] if i < n-1 else 129
            
            # We need two numbers x, y such that left_bound < x < y < right_bound
            # Or just two numbers in that range, sorted.
            candidates = [x for x in unused if left_bound < x < right_bound]
            if len(candidates) < 2:
                continue
            
            random.shuffle(candidates)
            
            # Try pairs
            # To save time, we can limit the number of pairs tried
            for idx_x in range(len(candidates)):
                x = candidates[idx_x]
                for idx_y in range(len(candidates)):
                    if idx_x == idx_y:
                        continue
                    y = candidates[idx_y]
                    
                    # Ensure order x < y
                    if x > y:
                        x, y = y, x
                    
                    # Check bounds again just in case (though filtered above)
                    if not (left_bound < x < y < right_bound):
                        continue
                        
                    new_seq = seq[:i] + [x, y] + seq[i+1:]
                    if is_valid(new_seq):
                        return new_seq
        return None

    best_seq = seq
    no_improve_count = 0
    
    while time.time() - start_time < timeout:
        improved = False
        
        # 1. Try to increase length (1-out/2-in)
        new_seq = try_1out_2in(seq)
        if new_seq is not None:
            seq = new_seq
            if len(seq) > len(best_seq):
                best_seq = seq
            improved = True
            no_improve_count = 0
            continue
        
        # 2. Try to insert (1-in) - might be possible if we didn't try all
        new_seq = try_insert(seq)
        if new_seq is not None:
            seq = new_seq
            if len(seq) > len(best_seq):
                best_seq = seq
            improved = True
            no_improve_count = 0
            continue
            
        # 3. Try to shuffle (1-out/1-in) to enable future moves
        new_seq = try_replace(seq)
        if new_seq is not None:
            seq = new_seq
            improved = True
            no_improve_count = 0
            continue
        
        # 4. If nothing worked, we might be stuck.
        # Try a random restart? Or just break?
        # Given the small search space, breaking is probably fine if we've explored well.
        # But let's try a few more random replacements to see if we can break out.
        no_improve_count += 1
        if no_improve_count > 20:
            # Try a more aggressive shuffle: replace 2 elements
            # Or just break
            break

    return best_seq
