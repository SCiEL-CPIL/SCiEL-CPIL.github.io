import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # Primes up to 128
    def is_prime(num):
        if num < 2: return False
        if num < 4: return True
        if num % 2 == 0 or num % 3 == 0: return False
        i = 5
        while i * i <= num:
            if num % i == 0 or num % (i + 2) == 0:
                return False
            i += 6
        return True

    primes = [p for p in range(2, 129) if is_prime(p)]
    composites = [c for c in range(2, 129) if not is_prime(c)]
    
    def get_initial_seq():
        return primes[:]

    def compute_products(seq):
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.add(p)
        return prods

    def try_insert(seq, prods, c, pos):
        """
        Try to insert c at pos in seq.
        Returns (new_seq, new_prods) if valid, else (None, None).
        """
        # Check increasing order
        if pos > 0 and seq[pos-1] >= c:
            return None, None
        if pos < len(seq) and c >= seq[pos]:
            return None, None
            
        n = len(seq)
        new_prods = set()
        
        # Generate new products involving c
        # Blocks ending at pos (0-indexed in new seq, so c is at pos)
        # In new seq, c is at index pos.
        # Blocks [i, pos] for i in 0..pos
        p = 1
        for i in range(pos, -1, -1):
            p *= seq[i]
            new_prods.add(p)
            
        # Blocks starting at pos
        # Blocks [pos, j] for j in pos..n-1
        # Note: [pos, pos] is just c, already added if i=pos in loop above?
        # In loop above, i=pos gives p=seq[pos]*c? No.
        # seq is old seq. c is not in it yet.
        # Let's be careful.
        
        # Let's reconstruct the logic.
        # Old seq: a_0 ... a_{n-1}
        # New seq: a_0 ... a_{pos-1}, c, a_pos ... a_{n-1}
        
        # New blocks are those that include c.
        # 1. Blocks ending at c: [i, pos] in new seq.
        #    Product = a_i * ... * a_{pos-1} * c
        # 2. Blocks starting at c: [pos, j] in new seq.
        #    Product = c * a_pos * ... * a_j
        # 3. Block [pos, pos] is c.
        
        # Let's calculate them.
        
        # Ending at c
        p = 1
        for i in range(pos, -1, -1):
            if i == pos:
                p = c
            else:
                p *= seq[i]
            new_prods.add(p)
            
        # Starting at c (excluding [pos, pos] which is c)
        p = c
        for j in range(pos, n):
            p *= seq[j]
            new_prods.add(p)
            
        # Check for duplicates within new_prods
        if len(new_prods) != 2 * pos + 1 + (n - pos) - 1:
             # Number of blocks ending at c: pos + 1
             # Number of blocks starting at c: n - pos
             # Total: pos + 1 + n - pos = n + 1
             # But [pos, pos] is counted in both?
             # Ending: [pos, pos] is c.
             # Starting: [pos, pos] is c.
             # So total unique blocks involving c is (pos + 1) + (n - pos) - 1 = n.
             # Wait.
             # Ending at c: indices i in 0..pos. Count pos+1.
             # Starting at c: indices j in pos..n-1. Count n-pos.
             # Intersection: [pos, pos].
             # Total = pos + 1 + n - pos - 1 = n.
             # So we expect n new products.
             pass # We use set, so duplicates are removed. We just need to check against old prods.
             
        # Check against old prods
        if new_prods & prods:
            return None, None
            
        # Valid
        new_seq = seq[:pos] + [c] + seq[pos:]
        new_prods = prods.union(new_prods)
        return new_seq, new_prods

    def greedy_solve(strategy):
        seq = get_initial_seq()
        prods = compute_products(seq)
        
        # Candidates to try
        # We can try composites in increasing order
        candidates = sorted(composites)
        
        for c in candidates:
            if time.time() > deadline - 5:
                break
                
            valid_positions = []
            for pos in range(len(seq) + 1):
                new_seq, new_prods = try_insert(seq, prods, c, pos)
                if new_seq is not None:
                    valid_positions.append((pos, new_seq, new_prods))
                    
            if valid_positions:
                # Choose best position based on strategy
                if strategy == 0: # Earliest
                    pos, seq, prods = valid_positions[0]
                elif strategy == 1: # Latest
                    pos, seq, prods = valid_positions[-1]
                elif strategy == 2: # Random
                    pos, seq, prods = random.choice(valid_positions)
                elif strategy == 3: # Min Max Product
                    # Heuristic: choose position that minimizes the largest new product
                    # This is expensive to calculate, so we skip or approximate
                    pos, seq, prods = valid_positions[0] # Fallback
                    
        return seq

    best_seq = []
    best_len = 0
    
    # Run multiple strategies
    for strat in range(4):
        if time.time() > deadline - 10:
            break
        seq = greedy_solve(strat)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
    # Random restarts with perturbed candidate order?
    # Since candidates are fixed 2..128, we can't change order easily.
    # But we can randomize the choice among valid positions more aggressively.
    
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        
        # Randomized greedy
        seq = get_initial_seq()
        prods = compute_products(seq)
        candidates = sorted(composites)
        
        for c in candidates:
            if time.time() > deadline - 2:
                break
            valid_positions = []
            for pos in range(len(seq) + 1):
                new_seq, new_prods = try_insert(seq, prods, c, pos)
                if new_seq is not None:
                    valid_positions.append((pos, new_seq, new_prods))
            if valid_positions:
                pos, seq, prods = random.choice(valid_positions)
                
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
    # Final polish: try to add any remaining numbers
    if time.time() < deadline - 2:
        used = set(best_seq)
        remaining = [c for c in range(2, 129) if c not in used]
        improved = True
        while improved:
            improved = False
            for c in remaining[:]:
                if time.time() > deadline - 1:
                    break
                for pos in range(len(best_seq) + 1):
                    new_seq, new_prods = try_insert(best_seq, compute_products(best_seq), c, pos)
                    if new_seq is not None:
                        best_seq = new_seq
                        used.add(c)
                        remaining.remove(c)
                        improved = True
                        break
                if improved:
                    break
                    
    return best_seq

# Run construct to ensure it works
# construct()
