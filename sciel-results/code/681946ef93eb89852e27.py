import random
import time
from collections import deque

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (evaluator kills at 90s)
    
    def find_cycle_of_length(k, adj):
        """Find a cycle of exactly length k in the graph. Returns list of vertices or None."""
        if k == N:
            # Hamiltonian cycle: use the base cycle
            return list(range(N))
        
        # For smaller k, use DFS with pruning
        # State: (current, path, visited_set)
        # We only need to find ONE cycle of length k
        
        if k <= 3:
            # Check all triangles
            for u in range(N):
                for v in adj[u]:
                    if v > u:
                        for w in adj[v]:
                            if w > v and w in adj[u]:
                                return [u, v, w]
            return None
        
        # DFS with backtracking, limit depth
        # Start from vertex 0 for determinism
        target_len = k
        
        def dfs(current, path, visited):
            if len(path) == target_len:
                if current in adj[path[0]]:
                    return path[:]
                return None
            if len(path) >= target_len:
                return None
            
            for nxt in adj[current]:
                if nxt not in visited:
                    if len(path) == target_len - 1:
                        # Last step: must connect back to start
                        if nxt == path[0]:
                            return path[:]
                    else:
                        visited.add(nxt)
                        result = dfs(nxt, path + [nxt], visited)
                        if result is not None:
                            return result
                        visited.remove(nxt)
            return None
        
        # Try from each starting vertex
        for start in range(N):
            visited = {start}
            result = dfs(start, [start], visited)
            if result is not None:
                return result
        return None
    
    def verify_pancyclic(adj, edges_set):
        """Verify all cycle lengths 3..40 exist. Returns dict of cycles or None."""
        cycles = {}
        for k in range(3, N + 1):
            cycle = find_cycle_of_length(k, adj)
            if cycle is None:
                return None
            # Verify the cycle is valid
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if (u, v) not in edges_set and (v, u) not in edges_set:
                    return None
            cycles[k] = cycle
        return cycles
    
    def build_adjacency(chords):
        adj = [set() for _ in range(N)]
        edges_set = set()
        # Base Hamiltonian cycle
        for i in range(N):
            j = (i + 1) % N
            adj[i].add(j)
            adj[j].add(i)
            edges_set.add((min(i,j), max(i,j)))
        # Chords
        for u, v in chords:
            adj[u].add(v)
            adj[v].add(u)
            edges_set.add((min(u,v), max(u,v)))
        return adj, edges_set
    
    # Start with a good initial construction
    # Base: Hamiltonian cycle
    # Chords: strategic placement to cover all cycle lengths
    
    # Strategy: Use chords that create cycles of specific lengths
    # Chord (i, j) in Hamiltonian cycle creates cycles of length (j-i) and (N - (j-i) + 2)
    # We want to cover lengths 3..39
    
    # Initial guess: chords from 0 to even vertices (covers odd cycle lengths)
    # Plus some chords for even lengths
    
    # Let's use a greedy approach:
    # 1. Start with Hamiltonian cycle
    # 2. For each missing cycle length, find a chord that creates it
    # 3. Minimize total chords
    
    # Precompute: for each pair (i,j), what cycle lengths does chord (i,j) create?
    def chord_cycle_lengths(i, j):
        """Return set of cycle lengths created by adding chord (i,j) to Hamiltonian cycle."""
        if i > j:
            i, j = j, i
        d = j - i
        if d == 0 or d == 1:
            return set()
        # Two cycles: one going i->i+1->...->j->i (length d+1)
        # and one going i->j->j+1->...->N-1->0->...->i (length N-d+1)
        lengths = set()
        lengths.add(d + 1)
        lengths.add(N - d + 1)
        return lengths
    
    # Greedy set cover: select minimum chords to cover all lengths 3..39
    needed = set(range(3, N))  # lengths 3..39
    all_chords = []
    for i in range(N):
        for j in range(i + 2, N):
            if j - i >= 2 and j - i <= N - 2:
                all_chords.append((i, j))
    
    # Greedy: pick chord covering most uncovered lengths
    chords = []
    covered = set()
    remaining = set(range(3, N))
    
    for _ in range(50):  # Max 50 chords
        if not remaining:
            break
        best_chord = None
        best_count = 0
        for i, j in all_chords:
            if (i, j) in chords:
                continue
            lengths = chord_cycle_lengths(i, j)
            new_covered = lengths & remaining
            if len(new_covered) > best_count:
                best_count = len(new_covered)
                best_chord = (i, j)
        if best_chord is None:
            break
        chords.append(best_chord)
        lengths = chord_cycle_lengths(best_chord[0], best_chord[1])
        remaining -= lengths
    
    # Initial solution
    initial_chords = chords[:]
    
    # Verify initial solution
    adj, edges_set = build_adjacency(initial_chords)
    cycles = verify_pancyclic(adj, edges_set)
    
    best_edges_count = len(edges_set)
    best_chords = initial_chords[:]
    best_cycles = cycles
    
    # Try to improve: remove chords if possible
    if cycles is not None:
        improved = True
        while improved:
            improved = False
            for idx in range(len(best_chords)):
                test_chords = best_chords[:idx] + best_chords[idx+1:]
                adj2, edges_set2 = build_adjacency(test_chords)
                cycles2 = verify_pancyclic(adj2, edges_set2)
                if cycles2 is not None:
                    best_chords = test_chords
                    best_cycles = cycles2
                    best_edges_count = len(edges_set2)
                    improved = True
                    break
    
    # Try random local search: swap chords
    if time.time() < deadline - 5:
        for iteration in range(2000):
            if time.time() > deadline - 2:
                break
            
            # Try replacing one chord with another
            if not best_chords:
                break
            idx = random.randint(0, len(best_chords) - 1)
            old_chord = best_chords[idx]
            
            # Try a random new chord
            for attempt in range(20):
                i = random.randint(0, N - 1)
                d = random.randint(2, N - 2)
                j = (i + d) % N
                if j == i or abs(i - j) == 1 or abs(i - j) == N - 1:
                    continue
                new_chord = (min(i, j), max(i, j))
                if new_chord == old_chord:
                    continue
                if new_chord in best_chords:
                    continue
                
                test_chords = best_chords[:idx] + [new_chord] + best_chords[idx+1:]
                adj2, edges_set2 = build_adjacency(test_chords)
                cycles2 = verify_pancyclic(adj2, edges_set2)
                if cycles2 is not None:
                    best_chords = test_chords
                    best_cycles = cycles2
                    best_edges_count = len(edges_set2)
                    break
    
    # Try adding chords and removing others
    if time.time() < deadline - 5:
        for iteration in range(1000):
            if time.time() > deadline - 2:
                break
            
            # Try adding a chord
            i = random.randint(0, N - 1)
            d = random.randint(2, N - 2)
            j = (i + d) % N
            if j == i or abs(i - j) == 1 or abs(i - j) == N - 1:
                continue
            new_chord = (min(i, j), max(i, j))
            if new_chord in best_chords:
                continue
            
            test_chords = best_chords + [new_chord]
            adj2, edges_set2 = build_adjacency(test_chords)
            cycles2 = verify_pancyclic(adj2, edges_set2)
            if cycles2 is not None:
                # Now try to remove some chords
                for remove_idx in range(len(test_chords)):
                    if remove_idx == len(test_chords) - 1:
                        continue  # Don't remove the newly added one
                    test2 = test_chords[:remove_idx] + test_chords[remove_idx+1:]
                    adj3, edges_set3 = build_adjacency(test2)
                    cycles3 = verify_pancyclic(adj3, edges_set3)
                    if cycles3 is not None:
                        best_chords = test2
                        best_cycles = cycles3
                        best_edges_count = len(edges_set3)
                        break
    
    # Final verification
    adj, edges_set = build_adjacency(best_chords)
    final_cycles = verify_pancyclic(adj, edges_set)
    
    if final_cycles is None:
        # Fallback: use a known valid construction
        # Star + path (the original seed)
        seed_edges = [(0, i) for i in range(1, N)] + [(i, i+1) for i in range(1, N-1)]
        seed_cycles = []
        for k in range(3, N + 1):
            if k == N:
                seed_cycles.append(list(range(N)))
            else:
                # Cycle: 0, 1, 2, ..., k-1, 0
                # This works because 0 is connected to all, and 1-2-...-(k-1) is a path
                seed_cycles.append(list(range(k)))
        return {'edges': seed_edges, 'cycles': seed_cycles}
    
    # Build final edges list
    final_edges = list(edges_set)
    # Sort for determinism
    final_edges.sort()
    
    # Build cycles list in order 3..40
    cycles_list = [final_cycles[k] for k in range(3, N + 1)]
    
    return {'edges': final_edges, 'cycles': cycles_list}
