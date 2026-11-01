import random
import time
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    New approach: Search over circulant graphs with carefully chosen connection sets,
    verify girth via fast non-backtracking walk enumeration, and attempt local repair
    for near-miss graphs.
    """
    if k == 3 and g == 16:
        start_time = time.time()
        total_timeout = 110.0
        
        # Strategy 1: Search circulant graphs C_N(a, b) where N is even
        # A circulant graph on N vertices has edges (i, i+a) and (i, i-b) mod N
        # This is 3-regular if gcd(N, a) = gcd(N, b) = 1 and a != b, a != -b mod N
        best_result = None
        
        # Try various N values starting from lower bound
        # Lower bound for n(3,16) is 512
        for N in range(512, 936, 2):  # Even N only
            if time.time() - start_time > total_timeout:
                break
            
            # For each N, try several (a, b) pairs
            # a and b should be coprime to N, and |a|, |b| should be large enough
            # to avoid short cycles
            candidates = generate_circulant_candidates(N)
            
            for a, b in candidates:
                if time.time() - start_time > total_timeout:
                    break
                
                # Quick check: ensure no immediate short cycles
                if has_short_cycle_fast(N, a, b, threshold=12):
                    continue
                
                # Build edges and verify girth
                edges = build_circulant_edges(N, a, b)
                girth_val = verify_girth_fast(N, edges, target_girth=g)
                
                if girth_val >= g:
                    return N, edges
                elif girth_val >= g - 2 and best_result is None:
                    best_result = (N, edges, girth_val)
        
        # Strategy 2: If we found a near-miss, try local repair
        if best_result:
            N, edges, girth_val = best_result
            if time.time() - start_time < total_timeout:
                repaired = local_repair_circulant(N, edges, target_girth=g, time_limit=total_timeout - (time.time() - start_time))
                if repaired:
                    final_girth = verify_girth_fast(N, repaired, target_girth=g)
                    if final_girth >= g:
                        return N, repaired
        
        # Strategy 3: Fall back to L-U-W construction
        return luw_construct(3, 16)
    else:
        return luw_construct(k, g)

def generate_circulant_candidates(N):
    """Generate promising (a, b) pairs for circulant graph C_N(a, b)."""
    candidates = []
    # a and b should be coprime to N
    # For 3-regular circulant: edges are (i, i+a) and (i, i-b) mod N
    # Need gcd(N, a) = 1, gcd(N, b) = 1, a != b, a != -b mod N
    # Also a != 0, b != 0
    
    # Try values that are "spread out" to avoid short cycles
    # Short cycles occur when small linear combinations of a and b are 0 mod N
    # So we want a and b such that |ma + nb| is large for small m, n
    
    # Heuristic: try a around N/3 and b around N/5, or other "incommensurate" values
    # Also try random pairs
    
    # Deterministic candidates based on number theory
    for a in range(1, N // 2 + 1):
        if _gcd(a, N) != 1:
            continue
        for b in range(1, N // 2 + 1):
            if _gcd(b, N) != 1:
                continue
            if a == b or a == N - b or a + b == N:
                continue
            # Check that small combinations don't wrap around too quickly
            # |a|, |b|, |a-b|, |a+b|, |2a-b|, |2b-a| should be "large" mod N
            min_combo = min(
                a % N, (N - a) % N,
                b % N, (N - b) % N,
                abs(a - b) % N, (N - abs(a - b)) % N,
                abs(a + b) % N, (N - abs(a + b)) % N
            )
            if min_combo < N // 10:
                continue
            candidates.append((a, b))
    
    # Add some random candidates for diversity
    for _ in range(50):
        a = random.randint(1, N - 1)
        b = random.randint(1, N - 1)
        if _gcd(a, N) == 1 and _gcd(b, N) == 1:
            if a != b and a != N - b and a + b != N:
                candidates.append((a, b))
    
    return candidates

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def has_short_cycle_fast(N, a, b, threshold=12):
    """Quick check if circulant C_N(a,b) has a cycle shorter than threshold."""
    # A cycle of length L corresponds to a sequence of steps in {a, -a, b, -b}
    # that sums to 0 mod N, with no backtracking (no consecutive same step)
    # For small L, we can enumerate all sequences
    
    # Non-backtracking walks of length L that return to start
    # Step options: +a, -a, +b, -b (but can't take same step consecutively)
    
    # For threshold=12, we check lengths 3 through 11
    # Actually, girth is the minimum cycle length, so we check if any cycle of length < threshold exists
    
    # Use BFS on the "state space" of (current_position, last_step)
    # State: (pos, last_dir) where last_dir is in {0, 1, 2, 3} for steps +a, -a, +b, -b
    # Start: (0, -1) meaning no last step
    # We want to find a path back to pos=0 with length >= 3 (simple cycle)
    
    # Actually, let's just check specific short cycle patterns
    # 3-cycle: a + a + a = 0 mod N? No, that would be 3a = 0 mod N, but gcd(a,N)=1 so N|3, unlikely for large N
    #          a + a + b = 0? 2a + b = 0 mod N
    #          a + b + b = 0? a + 2b = 0 mod N
    #          a + b + a = 0? 2a + b = 0 (same as above)
    #          etc.
    
    # Let's do a proper BFS for cycles up to length threshold-1
    steps = [(a, 0), (-a, 1), (b, 2), (-b, 3)]
    step_names = ['a', '-a', 'b', '-b']
    
    # BFS: state = (position, last_step_index)
    # We start from position 0 with no last step
    # We want to find a return to position 0 after >= 3 steps
    
    # visited[length][pos][last_step] = True if we've been here
    # But this could be large. For threshold=12, max length is 11, N up to 936, 4 last steps
    # That's 11 * 936 * 4 = ~41k states, manageable
    
    max_len = threshold - 1
    # visited[l][pos][last] = True
    visited = [[[False]*4 for _ in range(N)] for _ in range(max_len + 1)]
    
    # Queue: (length, position, last_step)
    queue = deque()
    # Start: length 0, position 0, last_step = -1 (special)
    # We'll handle the first step separately
    
    for step_idx, (step_val, _) in enumerate(steps):
        new_pos = step_val % N
        if new_pos == 0:
            return True  # Self-loop, but we checked gcd so this shouldn't happen
        visited[1][new_pos][step_idx] = True
        queue.append((1, new_pos, step_idx))
    
    while queue:
        length, pos, last_step = queue.popleft()
        if length >= max_len:
            continue
        
        for step_idx, (step_val, _) in enumerate(steps):
            if step_idx == last_step:
                continue  # No backtracking
            new_pos = (pos + step_val) % N
            if new_pos == 0:
                # Found a cycle of length length + 1
                if length + 1 >= 3:  # Simple cycle
                    return True
            if not visited[length + 1][new_pos][step_idx]:
                visited[length + 1][new_pos][step_idx] = True
                queue.append((length + 1, new_pos, step_idx))
    
    return False

def build_circulant_edges(N, a, b):
    """Build edge list for circulant graph C_N(a, b)."""
    edges = []
    seen = set()
    for i in range(N):
        for delta in [a, -b]:
            j = (i + delta) % N
            if i < j:
                edge = (i, j)
            else:
                edge = (j, i)
            if edge not in seen:
                seen.add(edge)
                edges.append(edge)
    return edges

def verify_girth_fast(N, edges, target_girth=16):
    """
    Verify girth using non-backtracking walk enumeration.
    Returns the girth, or early-exits if girth < target_girth.
    """
    # Build adjacency
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, do BFS in the non-backtracking state space
    # State: (current_vertex, previous_vertex)
    # We want to find the shortest cycle
    
    min_girth = float('inf')
    
    for start in range(N):
        # BFS from (start, -1) where -1 means no previous vertex
        # visited[length][current][prev] 
        # But we can use a dict for efficiency
        
        # Actually, let's use a different approach:
        # For each starting vertex, do a BFS tracking (current, prev)
        # The first time we return to start, that's a cycle
        
        # Use a queue of (current, prev, length)
        # visited is a set of (current, prev)
        
        visited = set()
        queue = deque()
        
        # Initialize: from start, go to each neighbor
        for neighbor in adj[start]:
            state = (neighbor, start)
            if state not in visited:
                visited.add(state)
                queue.append((neighbor, start, 1))
        
        while queue:
            current, prev, length = queue.popleft()
            
            if current == start and length >= 2:
                # Found a cycle of length length + 1
                cycle_len = length + 1
                if cycle_len < min_girth:
                    min_girth = cycle_len
                    if min_girth < target_girth:
                        return min_girth
                continue  # Don't expand from start
            
            for neighbor in adj[current]:
                if neighbor == prev:
                    continue  # No backtracking
                state = (neighbor, current)
                if state not in visited:
                    visited.add(state)
                    queue.append((neighbor, current, length + 1))
    
    return min_girth if min_girth != float('inf') else N

def local_repair_circulant(N, edges, target_girth, time_limit):
    """
    Attempt to repair a near-miss graph by 2-switches.
    Returns new edges if successful, else None.
    """
    start_time = time.time()
    
    # Build adjacency
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find a short cycle and try to break it
    # This is a simplified repair: try random 2-switches and check if girth improves
    
    current_edges = list(edges)
    current_girth = verify_girth_fast(N, current_edges, target_girth)
    
    if current_girth >= target_girth:
        return current_edges
    
    best_edges = current_edges
    best_girth = current_girth
    
    for _ in range(100):
        if time.time() - start_time > time_limit:
            break
        
        # Pick two random edges and do a 2-switch
        if len(current_edges) < 2:
            break
        
        i, j = random.sample(range(len(current_edges)), 2)
        u1, v1 = current_edges[i]
        u2, v2 = current_edges[j]
        
        # Check if 2-switch is valid (no loops or multi-edges)
        # Option 1: (u1, v1), (u2, v2) -> (u1, u2), (v1, v2)
        # Option 2: (u1, v1), (u2, v2) -> (u1, v2), (v1, u2)
        
        new_edges = None
        for option in [1, 2]:
            if option == 1:
                new_u1, new_v1 = u1, u2
                new_u2, new_v2 = v1, v2
            else:
                new_u1, new_v1 = u1, v2
                new_u2, new_v2 = v1, u2
            
            # Check validity
            if new_u1 == new_v1 or new_u2 == new_v2:
                continue
            if (min(new_u1, new_v1), max(new_u1, new_v1)) == (min(new_u2, new_v2), max(new_u2, new_v2)):
                continue
            
            # Check for multi-edges
            test_edges = list(current_edges)
            test_edges[i] = (min(new_u1, new_v1), max(new_u1, new_v1))
            test_edges[j] = (min(new_u2, new_v2), max(new_u2, new_v2))
            
            edge_set = set(test_edges)
            if len(edge_set) != len(test_edges):
                continue
            
            # Check degree
            deg = [0] * N
            for e in test_edges:
                deg[e[0]] += 1
                deg[e[1]] += 1
            if all(d == 3 for d in deg):
                new_edges = test_edges
                break
        
        if new_edges is None:
            continue
        
        new_girth = verify_girth_fast(N, new_edges, target_girth)
        
        if new_girth > best_girth:
            best_girth = new_girth
            best_edges = new_edges
            current_edges = new_edges
            
            if best_girth >= target_girth:
                return best_edges
    
    if best_girth >= target_girth:
        return best_edges
    return None

def luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q).
    """
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
