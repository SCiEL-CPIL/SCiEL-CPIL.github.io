import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    def get_primes(n):
        sieve = [True] * (n+1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5)+1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]
    
    def compute_all_products(seq):
        """Compute set of all nonempty consecutive block products."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return products
    
    def is_valid(seq):
        """Check if all consecutive block products are distinct."""
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
    
    def try_insert(seq, x):
        """Try to insert x into seq at the best valid position. Returns (new_seq, pos) or (None, -1)."""
        n = len(seq)
        # Find valid insertion positions: between seq[i-1] and seq[i] where seq[i-1] < x < seq[i]
        # Or at start if x < seq[0], or at end if x > seq[-1]
        valid_positions = []
        if x < seq[0]:
            valid_positions.append(0)
        for i in range(n):
            if seq[i] < x < seq[i+1] if i+1 < n else False:
                valid_positions.append(i+1)
        if x > seq[-1]:
            valid_positions.append(n)
        
        # Also check if x fits between any consecutive elements
        # Actually, for strictly increasing, x must be placed such that seq[pos-1] < x < seq[pos]
        # Let me redo this properly
        valid_positions = []
        if n == 0:
            valid_positions = [0]
        else:
            if x < seq[0]:
                valid_positions.append(0)
            for i in range(n-1):
                if seq[i] < x < seq[i+1]:
                    valid_positions.append(i+1)
            if x > seq[-1]:
                valid_positions.append(n)
        
        if not valid_positions:
            return None, -1
        
        # Try each valid position and check validity
        # To be efficient, try positions in order of "closeness" to maintain density
        # Actually, just try all valid positions
        for pos in valid_positions:
            if time.time() > deadline:
                return None, -1
            new_seq = seq[:pos] + [x] + seq[pos:]
            if is_valid(new_seq):
                return new_seq, pos
        return None, -1
    
    primes = get_primes(128)
    best_seq = primes[:]
    best_len = len(primes)
    
    # Get all composites in [2, 128]
    prime_set = set(primes)
    composites = [x for x in range(2, 129) if x not in prime_set]
    
    # Strategy 1: Greedy insertion - try to insert each composite
    # Try composites in different orders
    for order in ['ascending', 'descending', 'smallest_first']:
        if time.time() > deadline:
            break
        seq = primes[:]
        if order == 'ascending':
            candidates = composites[:]
        elif order == 'descending':
            candidates = composites[::-1]
        else:
            candidates = composites[:]
        
        for comp in candidates:
            if time.time() > deadline:
                break
            new_seq, pos = try_insert(seq, comp)
            if new_seq is not None:
                seq = new_seq
        
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
    
    # Strategy 2: Iterative refinement - try adding remaining composites
    remaining = [x for x in range(2, 129) if x not in set(best_seq)]
    for comp in remaining:
        if time.time() > deadline:
            break
        new_seq, pos = try_insert(best_seq, comp)
        if new_seq is not None:
            best_seq = new_seq
            best_len = len(best_seq)
    
    # Strategy 3: Local search - try removing one element and adding two
    for _ in range(3):  # Multiple passes
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(best_seq)):
            if time.time() > deadline:
                break
            # Try removing best_seq[i] and see if we can add two composites
            reduced = best_seq[:i] + best_seq[i+1:]
            remaining_after_removal = [x for x in range(2, 129) if x not in set(reduced)]
            # Try to add composites to reduced
            seq2 = reduced[:]
            for comp in remaining_after_removal:
                if time.time() > deadline:
                    break
                new_seq2, _ = try_insert(seq2, comp)
                if new_seq2 is not None:
                    seq2 = new_seq2
            if len(seq2) > len(best_seq):
                best_seq = seq2
                best_len = len(seq2)
                improved = True
                break
        if not improved:
            break
    
    # Final validation
    if not is_valid(best_seq):
        # Fallback to primes
        best_seq = primes[:]
    
    return best_seq
