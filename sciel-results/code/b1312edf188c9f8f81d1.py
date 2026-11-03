import random
import time
import sys
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Strategy: 
    1. Systematic search for cyclic (Z_n-regular) graphs with smart offset sampling.
    2. Fallback: Lazebnik-Ustimenko-Woldar (LW) graph with optimal q.
    """
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill
    
    def check_girth_cyclic_fast(n, offsets):
        """
        Check if the cyclic graph with given offsets has girth >= g.
        Uses BFS on (current_sum, last_offset_index) states.
        Returns True if girth >= g.
        Optimized with array-based visited tracking.
        """
        if len(offsets) != k:
            return False
        
        # Basic sanity checks
        if any(o % n == 0 for o in offsets):
            return False
        for i in range(k):
            for j in range(i + 1, k):
                if offsets[i] % n == offsets[j] % n:
                    return False
                if (offsets[i] + offsets[j]) % n == 0:
                    return False
        
        # State: (sum, last_idx) -> depth visited
        # Use a 2D list for O(1) lookup
        visited_depth = [[-1] * k for _ in range(n)]
        
        queue = deque()
        
        for i in range(k):
            s = offsets[i] % n
            state_idx = i
            visited_depth[s][state_idx] = 1
            queue.append((s, state_idx, 1))
        
        while queue:
            s, last_idx, depth = queue.popleft()
            if depth >= g:
                continue
            for j in range(k):
                if j == last_idx:
                    continue
                if (offsets[j] + offsets[last_idx]) % n == 0:
                    continue
                new_s = (s + offsets[j]) % n
                if new_s == 0:
                    if depth + 1 < g:
                        return False
                    continue
                if visited_depth[new_s][j] != -1:
                    continue
                visited_depth[new_s][j] = depth + 1
                queue.append((new_s, j, depth + 1))
        
        return True

    def make_cyclic_graph(n, offsets):
        """Create edge list for cyclic graph, verify it's simple and k-regular."""
        edges = []
        edge_set = set()
        for i in range(n):
            for off in offsets:
                j = (i + off) % n
                if i == j:
                    return None
                key = (min(i, j), max(i, j))
                if key in edge_set:
                    return None
                edge_set.add(key)
                edges.append((i, j))
        
        degree = [0] * n
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        if any(d != k for d in degree):
            return None
        
        return edges

    def valid_offsets(n, k, offsets):
        """Check if offsets form a valid connection set for a k-regular cyclic graph."""
        if len(offsets) != k:
            return False
        seen = set()
        for o in offsets:
            om = o % n
            if om in seen or om == 0:
                return False
            seen.add(om)
            seen.add((-om) % n)  # undirected: edge i->(i+o) same as (i-o)->i
        # No two offsets should be negatives of each other (would create double edges)
        # Actually for undirected cyclic graph, offsets {a, -a} creates each edge twice
        # We need the set {o, -o mod n} for each o to partition {1,...,n-1} appropriately
        # For k-regular: we need exactly k/2 pairs {a, -a} or k offsets where no o = -o mod n
        # and the union of {o, -o} covers distinct elements
        # Simplest: just check no duplicates and no zero, and verify degree in make_cyclic_graph
        return True

    def generate_random_offsets(n, k):
        """Generate random valid offsets for a k-regular cyclic graph."""
        # For undirected k-regular cyclic graph on Z_n:
        # We need k/2 pairs {a, -a} if k is even, or (k-1)/2 pairs plus one self-inverse
        # Self-inverse: o = -o mod n, i.e., 2o ≡ 0 mod n, i.e., o = n/2 (if n even)
        # For k=4: need 2 pairs {a,-a} and {b,-b}, or one pair + one self-inverse (if n even)
        
        candidates = list(range(1, n))
        random.shuffle(candidates)
        
        offsets = []
        used = set()
        
        for c in candidates:
            if c in used:
                continue
            # Check if adding c and -c (mod n) is valid
            neg = (-c) % n
            if c == neg:
                # Self-inverse: only valid if we can add it once
                # This creates edges i <-> (i + n/2), which is a matching
                # For k=4, we'd need 2 such self-inverse offsets, but only one exists (n/2)
                # So for k=4 with even n, we can have at most 1 self-inverse
                if n % 2 == 0 and c == n // 2:
                    offsets.append(c)
                    used.add(c)
                    if len(offsets) == k:
                        return offsets
                continue
            # Need both c and neg to be unused
            if neg in used:
                continue
            offsets.append(c)
            used.add(c)
            if neg not in used:
                offsets.append(neg)
                used.add(neg)
            if len(offsets) == k:
                return offsets
            if len(offsets) > k:
                return None
        
        return None if len(offsets) != k else offsets

    def generate_structured_offsets(n, k, base_offsets, perturb_range):
        """Perturb known good offsets to find nearby valid ones."""
        offsets = list(base_offsets)
        # Randomly perturb some offsets
        for _ in range(perturb_range):
            idx = random.randint(0, k - 1)
            delta = random.randint(1, max(1, n // 10))
            offsets[idx] = (offsets[idx] + delta) % n
            if offsets[idx] == 0:
                offsets[idx] = 1
        return offsets

    best_n = float('inf')
    best_edges = None

    # Search range: Moore bound to known best
    min_n = 243
    max_n_to_try = 320

    # Known good cyclic graphs for (k=4, g=10) if any exist
    # Let's try systematic search from small n upward
    
    # Phase 1: Try known/likely good n values with structured search
    # Phase 2: Broad random search
    
    # Known: For k=4, g=10, the smallest known is 320.
    # Let's search from 243 to 320.
    
    # First, try a few promising n values with many random attempts
    # Interleave different n values to avoid getting stuck
    
    search_order = list(range(min_n, max_n_to_try + 1))
    random.shuffle(search_order)
    # But we want smaller n first, so sort by n but with some randomization
    # Use a priority: smaller n gets more attempts
    
    attempts_per_n = {}
    
    for n in range(min_n, max_n_to_try + 1):
        if time.time() > deadline - 15:
            break
        
        # Number of attempts for this n: more for smaller n
        base_attempts = max(50, int(20000 / (n - min_n + 1)))
        
        found = False
        attempts = 0
        n_start = time.time()
        n_budget = min(5.0, max(1.0, (deadline - 15 - time.time()) / (max_n_to_try + 1 - n)))
        
        while time.time() < n_start + n_budget and time.time() < deadline - 15:
            attempts += 1
            
            # Generate random offsets
            offsets = generate_random_offsets(n, k)
            if offsets is None or len(offsets) != k:
                continue
            
            if check_girth_cyclic_fast(n, offsets):
                edges = make_cyclic_graph(n, offsets)
                if edges is not None:
                    best_n = n
                    best_edges = edges
                    found = True
                    break
        
        if found:
            break

    if best_edges is not None:
        return best_n, best_edges

    # Phase 2: If Phase 1 didn't find anything, try more targeted search
    # Focus on n values that are likely to work (e.g., multiples of certain numbers)
    for n in range(min_n, max_n_to_try + 1):
        if time.time() > deadline - 10:
            break
        
        # Try structured offsets based on known patterns
        base_offsets = [1, 2, 3, 4]  # Simple base
        for perturb in range(100):
            if time.time() > deadline - 10:
                break
            offsets = generate_structured_offsets(n, k, base_offsets, random.randint(1, 5))
            if offsets is None or len(offsets) != k:
                continue
            if check_girth_cyclic_fast(n, offsets):
                edges = make_cyclic_graph(n, offsets)
                if edges is not None:
                    best_n = n
                    best_edges = edges
                    return best_n, best_edges

    if best_edges is not None:
        return best_n, best_edges

    # Fallback: LW Graph
    def smallest_prime_power_at_least(target):
        p = 2
        while True:
            x = 1
            val = p
            while val < target:
                val *= p
                x += 1
            if val >= target:
                return val
            p += 1
            if p > 100:
                return 101

    q = smallest_prime_power_at_least(k - 1)
    if q < 2:
        q = 2
        
    d = (g + 2) // 2
    if d < 3:
        d = 3
        
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    elif q == 3:
        add = lambda x, y: (x + y) % 3
        sub = lambda x, y: (x - y) % 3
        mul = lambda x, y: (x * y) % 3
    elif q == 2:
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: x & y
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
