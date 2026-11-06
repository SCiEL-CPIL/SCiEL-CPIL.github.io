import random
import time
from itertools import islice

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    def check_validity(seq):
        """
        Check if seq is strictly increasing and all subarray products are distinct.
        Returns True if valid.
        """
        if not seq:
            return True
        if any(seq[i] >= seq[i+1] for i in range(len(seq)-1)):
            return False
        
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True

    def greedy_construct(randomize=False, k=1):
        values = []
        products = set()
        candidates = list(range(2, 129))
        
        for _ in range(len(candidates)):
            valid_candidates = []
            for x in candidates:
                if values and x <= values[-1]:
                    continue
                # Check if x can be appended
                # New products: x * suffix_prod for all suffixes of values
                # Suffixes of values: [values[-1], values[-2]*values[-1], ...]
                # Actually, we need to check against ALL existing products.
                
                # Optimization: 
                # 1. x not in products
                # 2. For all suffix products S of values, x*S not in products
                # Note: Suffix products are a subset of all products.
                # But we need to check x*S against ALL products, not just suffixes.
                
                if x in products:
                    continue
                
                # Calculate suffix products of values
                # suffix_prods = [values[-1], values[-2]*values[-1], ...]
                # We can compute these on the fly or maintain them.
                # For simplicity in this helper, we recompute or use a list.
                
                suffix_prod = 1
                valid = True
                # We need to check x * (product of values[i..end]) for all i
                # Let's iterate backwards
                prod = 1
                for a in reversed(values):
                    prod *= a
                    if x * prod in products:
                        valid = False
                        break
                if valid:
                    # Also need to ensure x*prod are distinct from each other?
                    # Yes, but since x > 1 and prod values are distinct (suffixes of a sequence with distinct products), 
                    # x*prod1 == x*prod2 => prod1 == prod2. So they are distinct.
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            if randomize:
                # Pick k random candidates from the smallest ones? 
                # Or just random?
                # Let's pick from the smallest k to keep it somewhat dense, or random.
                # The seed picked the smallest. Let's try picking randomly from valid.
                x = random.choice(valid_candidates)
            else:
                x = min(valid_candidates)
            
            values.append(x)
            
            # Update products
            # Add x
            # Add x * suffix_prods
            suffix_prod = 1
            # We need to add x, x*last, x*last*prev...
            # Actually, we can just add all new subarray products ending at x.
            prod = 1
            for a in reversed(values):
                prod *= a
                products.add(prod)
            
            candidates.remove(x)
        
        return values

    best_seq = greedy_construct(randomize=False)
    best_len = len(best_seq)
    
    # Local Search / Randomized Insertion
    for trial in range(1000):
        if time.time() > deadline:
            break
            
        # Initialize current sequence
        # Use a randomized greedy start
        current_seq = greedy_construct(randomize=True)
        
        # Try to improve by inserting
        improved = True
        while improved and time.time() < deadline:
            improved = False
            used = set(current_seq)
            unused = [x for x in range(2, 129) if x not in used]
            random.shuffle(unused)
            
            for u in unused:
                if time.time() > deadline:
                    break
                # Try inserting u at every valid position
                # Valid positions are where A[p-1] < u < A[p]
                # We can find the insertion point using bisect
                import bisect
                pos = bisect.bisect_left(current_seq, u)
                
                # Check pos and pos-1? 
                # If current_seq[pos-1] < u and (pos == len or u < current_seq[pos]), then pos is valid.
                # Since bisect_left gives the first index where A[i] >= u, 
                # if A[pos-1] < u, then u can be inserted at pos.
                # If A[pos] == u, it's already there (but u is unused, so A[pos] != u).
                # So if pos < len and A[pos] < u, then u > A[pos], so it can't be inserted at pos (would violate increasing order relative to next).
                # Wait. bisect_left(A, u) returns i such that A[0..i-1] < u <= A[i..n-1].
                # So A[i-1] < u. And u <= A[i].
                # If u == A[i], invalid (duplicate). But u is unused, so u != A[i].
                # So u < A[i].
                # Thus A[i-1] < u < A[i].
                # So the ONLY valid position is i = pos.
                
                if pos == len(current_seq):
                    # Check if u > current_seq[-1]
                    if current_seq and u <= current_seq[-1]:
                        continue
                else:
                    # Check if u < current_seq[pos]
                    if u >= current_seq[pos]:
                        continue
                    # And u > current_seq[pos-1] is guaranteed by bisect_left logic?
                    # Yes, bisect_left ensures A[pos-1] < u.
                
                # Try inserting at pos
                new_seq = current_seq[:pos] + [u] + current_seq[pos:]
                
                # Check validity
                # We only need to check new products involving u.
                # But for simplicity and safety, let's use the full check for small N.
                # N is ~35. Full check is O(N^2) ~ 1000 ops.
                # 127 unused * 1 pos * 1000 ops = 127k ops per iteration.
                # Very fast.
                
                if check_validity(new_seq):
                    current_seq = new_seq
                    if len(current_seq) > best_len:
                        best_seq = current_seq
                        best_len = len(current_seq)
                    improved = True
                    # Restart the while loop to try other insertions
                    break
        
        # If no improvement, we can try to remove an element and re-insert?
        # Or just rely on the next trial with a different greedy start.
        
    return best_seq
