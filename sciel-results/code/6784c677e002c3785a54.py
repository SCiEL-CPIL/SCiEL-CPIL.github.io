import random
import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    if k != 3 or g != 16:
        # Fallback for other parameters: use the seed LUW-like construction
        return _seed_construct(k, g)
    
    # Strategy: Search for a circulant graph C_N(a,b) with girth >= 16.
    # A circulant graph on Z_N with edges (i, i+a) and (i, i-b) is 3-regular.
    # Girth is determined by the shortest non-backtracking walk returning to 0.
    # We can check this efficiently by enumerating words in {a, b, -a, -b}
    # that avoid immediate backtracking (e.g., a then -a).
    
    # N must be even for a 3-regular graph? No, N*k must be even. 3*N even -> N even.
    # Also N must be > 2*max(a,b).
    
    # Heuristic search space:
    # Try N from a lower bound up to 936 (known best).
    # Moore bound for n(3,16) is 512.
    # Let's try N values that are "nice" or just scan a range.
    # Scanning all N from 512 to 936 is ~424 values.
    # For each N, we need to find a, b.
    # a, b < N/2.
    # This is O(N^2) per N, which is too slow if done naively.
    
    # Optimization:
    # We can iterate N. For each N, iterate a. For each a, iterate b.
    # Check girth.
    # To speed up girth check, we can precompute or use a fast BFS on the Cayley graph.
    # Since the graph is a Cayley graph of Z_N, the girth is the length of the shortest
    # non-trivial relation in the generators {a, b} with no immediate cancellation.
    
    # Let's implement a fast check:
    # BFS from 0.
    # State: (current_pos, prev_step). prev_step is the edge used to arrive.
    # If prev_step is None, we are at start.
    # We want to find if we can reach 0 with length L < 16, where the path is non-backtracking.
    # Actually, simpler:
    # Enumerate all non-backtracking walks of length L < 16.
    # If any sum to 0 mod N, girth <= L.
    # Number of such walks:
    # L=1: 0 (loops not allowed in simple graph, but a!=0, b!=0)
    # L=2: a-a, b-b, a-b, b-a. Sums: 0, a-b, b-a.
    #   If a==b or a==-b or a==0 or b==0, we have issues. We assume a!=b, a!=-b, a!=0, b!=0.
    #   So L=2 cycles only if a-b=0 (a=b, excluded) or b-a=0 (excluded).
    #   Wait, non-backtracking means we can't go a then -a.
    #   Walks of length 2:
    #   a, a -> 2a
    #   a, b -> a+b
    #   a, -b -> a-b
    #   b, a -> a+b
    #   b, b -> 2b
    #   b, -a -> b-a
    #   -a, a -> 0 (Backtracking! -a then a cancels. Is this allowed? No, "non-backtracking" usually forbids immediately reversing an edge. So -a followed by a is forbidden.)
    #   -a, b -> b-a
    #   -a, -b -> -a-b
    #   -b, a -> a-b
    #   -b, b -> 0 (Backtracking)
    #   -b, -a -> -a-b
    #   
    #   So valid non-backtracking walks of length 2:
    #   2a, a+b, a-b, 2b, b-a, -a-b.
    #   If any of these are 0 mod N, we have a cycle of length 2 (which is impossible in simple graph unless a=b or a=-b, which we exclude) or we just found a short cycle?
    #   Wait, if 2a = 0 mod N, that means a = N/2. Then the edge (0, N/2) is a double edge? No, (0, N/2) and (N/2, 0) are the same edge. But we have edges (i, i+a) and (i, i-b).
    #   If a = N/2, then (i, i+N/2) is an edge. Also (i, i-N/2) is the same edge.
    #   So if a = N/2, the graph has multiple edges or is not simple?
    #   Edges are undirected. {i, i+a}. If a = N/2, {i, i+N/2} = {i+N/2, i}.
    #   So the set of edges is just N/2 edges? No, i ranges 0..N-1.
    #   i=0 -> {0, N/2}
    #   i=N/2 -> {N/2, 0}
    #   So we have repeated edges. Not simple.
    #   So we must ensure a != N/2 and b != N/2.
    #   Also a != b, a != -b.
    
    # Let's write a function `has_short_cycle(N, a, b, max_len)` that returns True if there is a non-backtracking closed walk of length < max_len.
    # We can use BFS on the state space (pos, last_gen).
    # last_gen in {0, 1, 2, 3} representing a, b, -a, -b.
    # Start states: (a, 0), (b, 1), (-a, 2), (-b, 3).
    # Target: reach (0, any) with path length L.
    # If L < g, return True.
    
    def has_short_cycle(N, a, b, g_limit):
        # Normalize a, b to [0, N)
        a = a % N
        b = b % N
        if a == 0 or b == 0: return True
        if a == b or a == (N - b) % N: return True # Multi-edge or loop
        
        # Generators
        gens = [a, b, (N - a) % N, (N - b) % N]
        # Map gen to index
        gen_to_idx = {g: i for i, g in enumerate(gens)}
        
        # BFS
        # dist[pos][last_idx] = min length to reach pos having come from last_idx
        # We only care about length < g_limit.
        
        # Optimize: Since g_limit is small (16), we can just do a bounded BFS.
        # State: (pos, last_idx)
        # Start:
        # From 0, we can go to a (last=0), b (last=1), -a (last=2), -b (last=3).
        # These have length 1.
        
        # Queue: (pos, last_idx, length)
        # Visited: set of (pos, last_idx) with min length?
        # Since we want min length, standard BFS works.
        
        visited = set()
        q = deque()
        
        for i in range(4):
            pos = gens[i]
            # Check if pos == 0 (length 1 cycle, impossible for simple graph with a,b!=0)
            if pos == 0:
                return True
            q.append((pos, i, 1))
            visited.add((pos, i))
            
        while q:
            pos, last_idx, length = q.popleft()
            
            if length >= g_limit:
                continue
                
            # Try all next moves
            for next_idx in range(4):
                # Non-backtracking: next_idx != opposite of last_idx
                # Opposite of 0 (a) is 2 (-a)
                # Opposite of 1 (b) is 3 (-b)
                if (last_idx == 0 and next_idx == 2) or (last_idx == 2 and next_idx == 0):
                    continue
                if (last_idx == 1 and next_idx == 3) or (last_idx == 3 and next_idx == 1):
                    continue
                
                new_pos = (pos + gens[next_idx]) % N
                new_state = (new_pos, next_idx)
                
                if new_state in visited:
                    continue
                
                visited.add(new_state)
                new_length = length + 1
                
                if new_pos == 0:
                    # Found a cycle of length new_length
                    if new_length < g_limit:
                        return True
                    # If new_length >= g_limit, we don't care about longer ones for this check
                    # But we might need to continue if we haven't exhausted all shorter paths?
                    # BFS ensures we find shortest first.
                    # If we found a cycle of length L < g_limit, we return True.
                    # If we found a cycle of length L >= g_limit, it doesn't disqualify us.
                    # But we must ensure we didn't miss a shorter one.
                    # Since BFS is level-order, if we are at level `length`, all shorter paths have been processed.
                    # So if we hit 0 at `new_length`, it's the shortest for this state.
                    # However, we might hit 0 via another state later with a shorter total length?
                    # No, BFS explores all states of length L before L+1.
                    # So the first time we hit 0, it is the global minimum girth (for non-backtracking).
                    # Wait, we might hit 0 at length 5, but there's another path to 0 of length 4?
                    # No, BFS processes length 1, then 2, then 3...
                    # So if we are processing length 4, and we generate a move to 0 of length 5,
                    # we haven't finished processing all length 4 states yet?
                    # Actually, `q.popleft()` takes the shortest.
                    # So if we generate `new_pos == 0` with `new_length`, it is the shortest possible
                    # for that specific state transition.
                    # But could there be another state `(pos2, last2)` at the same level `length`
                    # that leads to 0 with `new_length` as well? Yes.
                    # Could there be a state at level `length-1` that leads to 0 with `length`?
                    # That would have been processed earlier.
                    # So, if we find `new_pos == 0` with `new_length < g_limit`, we can return True immediately.
                    # If `new_length >= g_limit`, we just don't add it to the queue (or add it but it won't matter for < g_limit).
                    # Actually, if `new_length >= g_limit`, we can skip adding to queue as we only care about < g_limit.
                    continue
                
                q.append((new_pos, next_idx, new_length))
                
        return False

    # Search loop
    # N must be even.
    # Try N from 512 to 936.
    # To save time, we can be smart about a, b.
    # Often, a and b are small or related to N.
    # Let's try a few specific patterns or random samples if full search is too slow.
    # Full search: N up to 936.
    # For each N, a from 1 to N/2. b from 1 to N/2.
    # ~400 * 400 * 400 = 64M checks.
    # Each check is BFS up to depth 15.
    # This is too slow in Python.
    
    # Optimization:
    # 1. Pre-filter a, b.
    # 2. Parallelize? No.
    # 3. Use C extension? No.
    # 4. Reduce search space.
    #    Known constructions often use a = 1, b = k or similar.
    #    Or a, b are quadratic residues.
    
    # Let's try a restricted search first:
    # 1. Try a=1, b=2..N/2.
    # 2. Try a=2, b=3..N/2.
    # 3. Try random samples.
    
    best_n = None
    best_edges = None
    
    # Time budget per N
    time_per_n = 0.05 # 50ms
    
    for N in range(512, 937, 2): # Even N
        if time.time() > deadline - 5.0:
            break
            
        n_start = time.time()
        found = False
        
        # Subset 1: a=1
        for b in range(2, N//2):
            if time.time() - n_start > time_per_n:
                break
            if has_short_cycle(N, 1, b, g):
                continue
            # Construct edges
            edges = []
            for i in range(N):
                edges.append((i, (i + 1) % N))
                edges.append((i, (i - b) % N))
            # Check simple
            # (i, i+1) and (i, i-b)
            # If b=1, multi-edge. We skipped b=1.
            # If b=N-1, i-b = i+1. We have b < N/2 so b != N-1.
            # If b = N/2, we skipped.
            # So edges are distinct?
            # (i, i+1) is unique for each i? No, (i, i+1) and (i+1, i+2) are different.
            # But (i, i+1) and (i+1, i) are the same undirected edge.
            # Our list has (i, i+1). Next i is i+1, edge (i+1, i+2).
            # So no duplicates in the list?
            # Wait, (i, i+1) is added.
            # Is (i+1, i) added? No, we only add (i, i+1) and (i, i-b).
            # So the edge set is { {i, i+1}, {i, i-b} for i in 0..N-1 }.
            # Are {i, i+1} and {j, j+1} distinct?
            # {i, i+1} = {j, j+1} implies i=j or i=j+1.
            # If i=j, same.
            # If i=j+1, {j+1, j+2} = {j, j+1} -> j+2=j -> 2=0 mod N -> N=2. N>2.
            # So {i, i+1} are distinct.
            # Same for {i, i-b}.
            # Can {i, i+1} = {j, j-b}?
            # i=j and i+1=j-b -> 1=-b -> b=N-1. (Excluded)
            # i=j-b and i+1=j -> 1=b. (Excluded b=1)
            # So edges are distinct.
            
            found = True
            break
            
        if found:
            best_n = N
            best_edges = edges
            break # Found the smallest N
            
        # Subset 2: a=2
        for b in range(3, N//2):
            if time.time() - n_start > time_per_n:
                break
            if has_short_cycle(N, 2, b, g):
                continue
            edges = []
            for i in range(N):
                edges.append((i, (i + 2) % N))
                edges.append((i, (i - b) % N))
            found = True
            best_n = N
            best_edges = edges
            break
            
        if found:
            break
            
        # Subset 3: Random sampling
        # If we haven't found it with a=1 or a=2, try some random pairs.
        # This is a gamble, but might hit a valid (a,b) if it exists but requires large a.
        # However, if a=1,2 fail, it's likely no small a works.
        # Given time constraints, we might skip this or do very few.
        
    if best_n is None:
        # Fallback to LUW construction (seed)
        return _seed_construct(k, g)
        
    return best_n, best_edges

def _seed_construct(k, g):
    # Copy of the original seed logic to ensure fallback validity
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
