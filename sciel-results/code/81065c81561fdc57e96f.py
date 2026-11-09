import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    def popcount(x):
        return bin(x).count('1')

    def to_result(n, adj_masks):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj_masks[u] & (1 << v):
                    edges.append([u, v])
        return {'n': int(n), 'edges': edges}

    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    def generate_paley(n):
        """Generate a Paley-type graph on n vertices."""
        candidate = n
        while True:
            if candidate % 4 == 1 and is_prime(candidate):
                p = candidate
                break
            candidate += 1
        
        residues = set()
        for i in range(1, p):
            residues.add((i * i) % p)
        
        adj_masks = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                diff = (u - v) % p
                if diff in residues:
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
        return adj_masks

    def generate_random(n, p=0.5):
        """Generate a random graph on n vertices."""
        adj_masks = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
        return adj_masks

    def count_violations(adj_masks, n, max_k=None):
        """Count regular induced subgraphs of size >= 6.
        Returns (count, list_of_violating_masks_up_to_10).
        Uses bitmask for fast degree computation.
        """
        if max_k is None:
            max_k = n
        
        count = 0
        violations = []
        
        for k in range(6, min(max_k, n) + 1):
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                
                first_deg = -1
                is_reg = True
                for v in comb:
                    deg = popcount(adj_masks[v] & mask)
                    if first_deg == -1:
                        first_deg = deg
                    elif deg != first_deg:
                        is_reg = False
                        break
                if is_reg:
                    count += 1
                    if k <= 10 and len(violations) < 50:
                        violations.append((mask, k))
        
        return count, violations

    def find_violation(adj_masks, n, max_k=None):
        """Find one regular induced subgraph of size >= 6.
        Returns (mask, k) or (None, None).
        """
        if max_k is None:
            max_k = n
        
        for k in range(6, min(max_k, n) + 1):
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                
                first_deg = -1
                is_reg = True
                for v in comb:
                    deg = popcount(adj_masks[v] & mask)
                    if first_deg == -1:
                        first_deg = deg
                    elif deg != first_deg:
                        is_reg = False
                        break
                if is_reg:
                    return mask, k
        return None, None

    def validate_full(adj_masks, n):
        """Check all k >= 6. Returns True if valid (no regular subgraph of size >= 6)."""
        mask, k = find_violation(adj_masks, n)
        return mask is None

    def local_search(adj_masks, n, max_steps, time_limit):
        """Local search to eliminate regular induced subgraphs.
        Uses bitmask for fast violation finding and repair.
        """
        adj_masks = adj_masks[:]
        current_count, _ = count_violations(adj_masks, n, max_k=10)
        best_count = current_count
        best_adj = adj_masks[:]
        
        for step in range(max_steps):
            if time.time() > time_limit:
                break
            
            # Find a violation
            mask, k = find_violation(adj_masks, n, max_k=10)
            if mask is None:
                # No violation up to k=10, do full check
                mask_full, k_full = find_violation(adj_masks, n)
                if mask_full is None:
                    return adj_masks, True
                else:
                    mask, k = mask_full, k_full
            
            # Extract vertices from mask
            vertices = []
            temp = mask
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                vertices.append(v)
                temp ^= lsb
            
            if len(vertices) < 2:
                continue
            
            # Try flipping edges to break the violation
            # Strategy: try all pairs in the violating set, pick the one
            # that most reduces violation count
            best_flip = None
            best_new_count = current_count + 1  # worse than current
            
            # Sample some pairs if the set is large
            if len(vertices) > 8:
                pairs = random.sample(list(itertools.combinations(vertices, 2)), min(20, len(vertices) * (len(vertices) - 1) // 2))
            else:
                pairs = list(itertools.combinations(vertices, 2))
            
            for u, v in pairs:
                # Try flipping this edge
                adj_masks[u] ^= (1 << v)
                adj_masks[v] ^= (1 << u)
                
                # Quick count (only up to k=10 for speed)
                new_count, _ = count_violations(adj_masks, n, max_k=10)
                
                # Undo flip
                adj_masks[u] ^= (1 << v)
                adj_masks[v] ^= (1 << u)
                
                if new_count < best_new_count:
                    best_new_count = new_count
                    best_flip = (u, v)
            
            if best_flip is None:
                # No improvement found, try a random flip
                u, v = random.sample(vertices, 2)
                adj_masks[u] ^= (1 << v)
                adj_masks[v] ^= (1 << u)
                current_count, _ = count_violations(adj_masks, n, max_k=10)
            else:
                u, v = best_flip
                adj_masks[u] ^= (1 << v)
                adj_masks[v] ^= (1 << u)
                current_count = best_new_count
            
            if current_count < best_count:
                best_count = current_count
                best_adj = adj_masks[:]
        
        # Final full validation on best
        if best_count == 0:
            if validate_full(best_adj, n):
                return best_adj, True
        
        # Also check current
        if current_count == 0:
            if validate_full(adj_masks, n):
                return adj_masks, True
        
        return best_adj, False

    def try_n(n, max_restarts, max_steps_per_restart):
        """Try to find a valid graph on n vertices using multiple strategies."""
        if time.time() > deadline - 15:
            return None
        
        init_strategies = []
        
        # 1. Paley initialization
        try:
            adj_paley = generate_paley(n)
            init_strategies.append(adj_paley)
        except Exception:
            pass
        
        # 2. Random initializations with different densities
        for p in [0.5, 0.4, 0.6, 0.3, 0.7]:
            adj_rand = generate_random(n, p)
            init_strategies.append(adj_rand)
        
        # 3. More random if time allows
        while len(init_strategies) < max_restarts and time.time() < deadline - 20:
            adj_rand = generate_random(n, random.uniform(0.3, 0.7))
            init_strategies.append(adj_rand)
        
        for init_adj in init_strategies:
            if time.time() > deadline - 15:
                break
            
            adj_masks, success = local_search(init_adj, n, max_steps_per_restart, deadline - 10)
            
            if success:
                # Final validation
                if validate_full(adj_masks, n):
                    return adj_masks
        
        return None

    # Fallback: small valid graph
    n_fallback = 5
    adj_fallback = [0] * n_fallback
    best_result = to_result(n_fallback, adj_fallback)
    
    # Try n values in descending order
    # 21 is the target for improvement. 20 is the known baseline.
    for n_target in [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 15:
            break
        
        # Adjust restarts and steps based on n
        if n_target >= 20:
            max_restarts = 15
            max_steps = 80
        elif n_target >= 18:
            max_restarts = 20
            max_steps = 120
        else:
            max_restarts = 30
            max_steps = 200
        
        adj = try_n(n_target, max_restarts=max_restarts, max_steps_per_restart=max_steps)
        if adj is not None:
            best_result = to_result(n_target, adj)
            break
    
    return best_result

if __name__ == "__main__":
    res = construct()
    print(res['n'])
