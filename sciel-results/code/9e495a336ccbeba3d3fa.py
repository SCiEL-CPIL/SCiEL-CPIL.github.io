import sys
import time
import heapq
from random import Random

def construct():
    sys.setrecursionlimit(100000)
    start_time = time.time()
    TOTAL_BUDGET = 140.0  # Safety margin below 150s kill

    def instance(splits, seed, palette=5):
        """
        Generates the graph instance as described in the task.
        """
        rng = Random(seed)
        # Cube faces: 6 faces, 4 vertices each
        # Vertices 0-7
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = set()
        for f in faces:
            for i in range(4):
                a, b = f[i], f[(i + 1) % 4]
                edges.add((min(a, b), max(a, b)))
        
        n = 8
        for _ in range(splits):
            # Pick a random face
            face_idx = rng.randrange(len(faces))
            face = faces.pop(face_idx)
            
            # Create 4 new vertices
            inner = list(range(n, n + 4))
            n += 4
            
            # Connect inner vertices to face vertices and to each other
            for i in range(4):
                a = face[i]
                b = face[(i + 1) % 4]
                in_i = inner[i]
                in_next = inner[(i + 1) % 4]
                
                # Edge between face vertex and inner vertex
                edges.add((min(a, in_i), max(a, in_i)))
                # Edge between inner vertices
                edges.add((min(in_i, in_next), max(in_i, in_next)))
                
                # Create new face: a, b, in_next, in_i
                faces.append([a, b, in_next, in_i])
            
            # Create the inner face
            faces.append(inner)
        
        # Generate random lists of 3 colors from 0..4 for each vertex
        lists = []
        for _ in range(n):
            colors = rng.sample(range(palette), 3)
            lists.append(colors)
            
        return n, sorted(edges), lists

    def solve_instance(n, edges, lists, time_limit):
        """
        Solves the list coloring problem using DSATUR + Forward Checking.
        Returns a list of colors or None if it fails or times out.
        """
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initial domains
        domains = [list(l) for l in lists]
        color = [-1] * n
        uncolored = set(range(n))
        
        # DSATUR tracking
        # sat[v] = set of colors used by neighbors
        sat = [set() for _ in range(n)]
        
        deadline = time.time() + time_limit
        state = {'ok': False, 'count': 0}

        def forward_check(v, c):
            """
            Removes color c from domains of uncolored neighbors of v.
            Returns (success, trail) where trail allows undoing.
            """
            trail = []
            for u in adj[v]:
                if u in uncolored:
                    if c in domains[u]:
                        domains[u].remove(c)
                        trail.append((u, c))
                        if not domains[u]:
                            return False, trail
            return True, trail

        def undo(trail):
            for u, c in trail:
                domains[u].append(c)

        def backtrack():
            if not uncolored:
                state['ok'] = True
                return True

            state['count'] += 1
            # Time check every 100 nodes
            if (state['count'] & 127) == 0 and time.time() > deadline:
                return False

            # DSATUR: Pick vertex with max saturation, then min domain size, then max degree
            best_v = -1
            best_key = None
            
            for v in uncolored:
                sat_size = len(sat[v])
                dom_size = len(domains[v])
                deg = len(adj[v])
                
                # Key: (sat_size, -dom_size, -deg) -> We want max sat, min dom, max deg
                # Heap uses min, so we negate what we want to maximize? 
                # Actually, let's just find the best manually or use a priority queue properly.
                # For simplicity and speed in Python, a linear scan over uncolored set is fine for n~400.
                # n = 8 + 4*splits. Max splits=100 -> n=408. Linear scan is O(n).
                
                if best_key is None:
                    best_v = v
                    best_key = (sat_size, -dom_size, -deg)
                else:
                    current_key = (sat_size, -dom_size, -deg)
                    # We want to maximize sat_size, so larger is better.
                    # We want to minimize dom_size, so smaller is better (hence -dom_size in key? No, let's sort properly).
                    # Let's define a score. Higher score is better.
                    # Score = sat_size * 10000 - dom_size * 100 + deg
                    score = sat_size * 10000 - dom_size * 100 + deg
                    if best_key is None:
                        best_key = score
                    elif score > best_key:
                        best_key = score
                        best_v = v

            if best_v == -1:
                return False

            uncolored.discard(best_v)
            
            # Sort colors in domain to try most constrained first? 
            # Or just iterate.
            for c in domains[best_v][:]:
                color[best_v] = c
                
                # Update saturation for neighbors
                for u in adj[best_v]:
                    if u in uncolored:
                        sat[u].add(c)
                
                success, trail = forward_check(best_v, c)
                
                if success:
                    if backtrack():
                        return True
                
                # Undo forward check
                undo(trail)
                
                # Undo saturation
                for u in adj[best_v]:
                    if u in uncolored:
                        sat[u].discard(c)
                        
            color[best_v] = -1
            uncolored.add(best_v)
            return False

        if backtrack() and state['ok']:
            # Verify solution
            for v in range(n):
                if color[v] not in lists[v]:
                    return None
                for u in adj[v]:
                    if color[v] == color[u]:
                        return None
            return [int(c) for c in color]
        
        return None

    out = []
    instances_params = [(20, 701), (40, 702), (60, 703), (100, 704)]
    
    for splits, seed in instances_params:
        remaining = TOTAL_BUDGET - (time.time() - start_time)
        if remaining < 1.0:
            # Not enough time, break
            break
            
        n, edges, lists = instance(splits, seed)
        
        # Allocate time for this instance
        # We have 4 instances. Let's give each a fair share, but prioritize solving.
        # If we are fast, we use less.
        time_limit = max(0.5, min(20.0, remaining * 0.4))
        
        res = solve_instance(n, edges, lists, time_limit)
        
        if res is not None:
            out.append(res)
        else:
            # Fallback: Greedy coloring (might be invalid, but keeps structure)
            # We need a valid structure. If we can't solve it, we return a list of valid colors from the list,
            # even if adjacent vertices clash. The score will be 0 for this instance, but the result is valid.
            # Greedy: Order by degree, pick first available color.
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            order = sorted(range(n), key=lambda x: -len(adj[x]))
            greedy_color = [-1] * n
            for v in order:
                used = set()
                for u in adj[v]:
                    if greedy_color[u] != -1:
                        used.add(greedy_color[u])
                c = -1
                for col in lists[v]:
                    if col not in used:
                        c = col
                        break
                if c == -1:
                    # If all colors used, pick the first one (will cause conflict, but valid type)
                    c = lists[v][0]
                greedy_color[v] = c
            
            out.append([int(c) for c in greedy_color])

    # Ensure we have exactly 4 results
    while len(out) < 4:
        # If we ran out of time, we need to generate dummy valid structures for remaining
        # This shouldn't happen with the time budget, but just in case.
        # We need to know n for the missing instance.
        pass

    # Final validation and type fixing
    # Re-generate instances to check lengths if out is shorter? 
    # Actually, if we break early, we don't have results for the later instances.
    # We must return 4 lists.
    
    # Let's re-iterate to ensure we have 4 entries.
    if len(out) < 4:
        # This implies we timed out or failed to generate.
        # We need to fill the rest with valid structures.
        # We need to know which ones are missing.
        # Since we iterate in order, if len(out) is k, instances 0..k-1 are done.
        # We need to generate instances k..3.
        
        for i in range(len(out), 4):
            splits, seed = instances_params[i]
            n, edges, lists = instance(splits, seed)
            # Use greedy as fallback
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            order = sorted(range(n), key=lambda x: -len(adj[x]))
            greedy_color = [-1] * n
            for v in order:
                used = set()
                for u in adj[v]:
                    if greedy_color[u] != -1:
                        used.add(greedy_color[u])
                c = -1
                for col in lists[v]:
                    if col not in used:
                        c = col
                        break
                if c == -1:
                    c = lists[v][0]
                greedy_color[v] = c
            out.append([int(c) for c in greedy_color])

    # Final type check
    for i in range(4):
        if not isinstance(out[i], list):
            out[i] = list(out[i])
        out[i] = [int(x) for x in out[i]]
        
    return out

if __name__ == "__main__":
    construct()
