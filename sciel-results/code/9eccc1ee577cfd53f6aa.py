import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    MAX_L = 100000
    
    # Segment tree for range minimum queries
    class SegTree:
        def __init__(self, n):
            self.n = n
            self.size = 1
            while self.size < n:
                self.size *= 2
            self.tree = [0] * (2 * self.size)
        
        def build(self, arr):
            for i in range(self.n):
                self.tree[self.size + i] = arr[i]
            for i in range(self.size - 1, 0, -1):
                self.tree[i] = min(self.tree[2*i], self.tree[2*i+1])
        
        def update(self, pos, val):
            i = self.size + pos
            self.tree[i] = val
            i //= 2
            while i >= 1:
                self.tree[i] = min(self.tree[2*i], self.tree[2*i+1])
                i //= 2
        
        def query_min(self, l, r):
            """Query min in [l, r)"""
            l += self.size
            r += self.size
            res = float('inf')
            while l < r:
                if l & 1:
                    res = min(res, self.tree[l])
                    l += 1
                if r & 1:
                    r -= 1
                    res = min(res, self.tree[r])
                l //= 2
                r //= 2
            return res
        
        def find_first_less_than(self, val, l=0, r=None):
            """Find first position in [l, r) where tree value < val"""
            if r is None:
                r = self.n
            if l >= r:
                return -1
            return self._find_first_less_than(1, 0, self.size, l, r, val)
        
        def _find_first_less_than(self, node, node_l, node_r, q_l, q_r, val):
            if node_r <= q_l or node_l >= q_r:
                return -1
            if self.tree[node] >= val:
                return -1
            if node_r - node_l == 1:
                return node_l if node_l < self.n else -1
            mid = (node_l + node_r) // 2
            left_res = self._find_first_less_than(2*node, node_l, mid, q_l, q_r, val)
            if left_res != -1:
                return left_res
            return self._find_first_less_than(2*node+1, mid, node_r, q_l, q_r, val)
    
    # Greedy initialization
    GREEDY_WINDOW = 3000
    residues = [0] * n_primes
    counts_greedy = np.zeros(GREEDY_WINDOW + 1, dtype=np.int32)
    
    for idx in range(n_primes):
        p = primes[idx]
        deficient_mask = counts_greedy[1:GREEDY_WINDOW + 1] < 2
        if not np.any(deficient_mask):
            residues[idx] = 1
        else:
            positions = np.arange(1, GREEDY_WINDOW + 1)
            def_pos = positions[deficient_mask]
            mods = def_pos % p
            res_counts = np.bincount(mods, minlength=p)
            residues[idx] = int(np.argmax(res_counts))
        
        r = residues[idx]
        r_eff = r if r > 0 else p
        if r_eff <= GREEDY_WINDOW:
            counts_greedy[r_eff::p] += 1
    
    # Build full counts and segment tree
    counts = np.zeros(MAX_L + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        start = r if r > 0 else p
        if start <= MAX_L:
            counts[start::p] += 1
    
    seg = SegTree(MAX_L + 1)
    seg.build(counts)
    
    def calc_L(seg_tree, current_L=None):
        """Find largest L such that all positions 1..L have count >= 2"""
        # Find first position where count < 2
        first_def = seg_tree.find_first_less_than(2, 1, MAX_L + 1)
        if first_def == -1:
            return MAX_L
        return first_def - 1
    
    current_L = calc_L(seg)
    best_L = current_L
    best_residues = residues.copy()
    
    # Local search with segment tree
    # For each prime, try changing residue and see if L improves
    # Key: only need to recompute L, which is O(log MAX_L) with seg tree
    
    hc_deadline = start_time + 50.0
    improved = True
    while improved and time.time() < hc_deadline:
        improved = False
        for i in range(n_primes):
            p = primes[i]
            old_r = residues[i]
            
            # Remove old residue
            start_old = old_r if old_r > 0 else p
            if start_old <= MAX_L:
                for pos in range(start_old, MAX_L + 1, p):
                    seg.update(pos, counts[pos] - 1)
                counts[start_old::p] -= 1
            
            best_r_local = old_r
            best_L_local = current_L
            
            # Try a subset of residues (not all, for speed)
            # Focus on residues that cover positions near current_L
            if p <= 50:
                candidates = list(range(p))
            else:
                # Sample residues that cover positions near the bottleneck
                candidates = list(range(min(p, 100)))
                # Also add some random ones
                for _ in range(50):
                    candidates.append(np.random.randint(p))
                candidates = list(set(candidates))
            
            for new_r in candidates:
                if new_r == old_r:
                    continue
                
                start_new = new_r if new_r > 0 else p
                if start_new > MAX_L:
                    continue
                
                # Add new residue
                for pos in range(start_new, MAX_L + 1, p):
                    seg.update(pos, counts[pos] + 1)
                counts[start_new::p] += 1
                
                new_L = calc_L(seg)
                
                if new_L > best_L_local:
                    best_L_local = new_L
                    best_r_local = new_r
                
                # Remove new residue
                for pos in range(start_new, MAX_L + 1, p):
                    seg.update(pos, counts[pos] - 1)
                counts[start_new::p] -= 1
            
            # Restore old residue
            if start_old <= MAX_L:
                for pos in range(start_old, MAX_L + 1, p):
                    seg.update(pos, counts[pos] + 1)
                counts[start_old::p] += 1
            
            if best_r_local != old_r:
                residues[i] = best_r_local
                current_L = best_L_local
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
                improved = True
    
    # Simulated annealing with segment tree
    T = 2.0
    cooling = 0.9998
    
    while time.time() < deadline:
        i = np.random.randint(n_primes)
        p = primes[i]
        old_r = residues[i]
        new_r = np.random.randint(p)
        
        if new_r == old_r:
            continue
        
        start_old = old_r if old_r > 0 else p
        start_new = new_r if new_r > 0 else p
        
        # Remove old
        if start_old <= MAX_L:
            for pos in range(start_old, MAX_L + 1, p):
                seg.update(pos, counts[pos] - 1)
            counts[start_old::p] -= 1
        
        # Add new
        if start_new <= MAX_L:
            for pos in range(start_new, MAX_L + 1, p):
                seg.update(pos, counts[pos] + 1)
            counts[start_new::p] += 1
        
        new_L = calc_L(seg)
        
        if new_L >= current_L:
            residues[i] = new_r
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            delta = new_L - current_L
            if delta > -200 * T:
                prob = np.exp(delta / T)
            else:
                prob = 0.0
            
            if np.random.random() < prob:
                residues[i] = new_r
                current_L = new_L
            else:
                # Revert
                residues[i] = old_r
                if start_new <= MAX_L:
                    for pos in range(start_new, MAX_L + 1, p):
                        seg.update(pos, counts[pos] - 1)
                    counts[start_new::p] -= 1
                if start_old <= MAX_L:
                    for pos in range(start_old, MAX_L + 1, p):
                        seg.update(pos, counts[pos] + 1)
                    counts[start_old::p] += 1
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    return best_residues
