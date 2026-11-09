import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_cycle(adj, start, max_len=50, rng=None):
        """Find a cycle starting and ending at start using DFS with Warnsdorff heuristic."""
        if rng is None:
            rng = random.Random()
        
        # Try multiple random orderings of neighbors
        best_cycle = None
        
        for attempt in range(20):
            path = [start]
            visited = {start}
            current = start
            
            while True:
                neighbors = [n for n in adj[current] if n not in visited]
                
                # Check if we can close the cycle
                if len(path) >= 3 and start in adj[current]:
                    cycle = path + [start]
                    if best_cycle is None or len(cycle) > len(best_cycle):
                        best_cycle = cycle
                    # We found a cycle, but let's see if we can find a longer one
                    # by continuing? No, the cycle is closed. 
                    # Actually, for decomposition, we want to remove a cycle.
                    # A longer cycle is better.
                    # Let's try to extend if possible? No, the edge (current, start) is used.
                    # So this is a valid cycle.
                    
                    # Try to find a longer one by not closing yet?
                    # We can only close if we go back to start.
                    # If we don't close, we continue.
                    # But we must eventually return to start to form a simple cycle in this DFS.
                    
                    # Let's break and return this cycle, or try to find a better one in next attempt.
                    break
                
                if not neighbors:
                    break
                
                # Warnsdorff: choose neighbor with fewest unvisited neighbors
                neighbors.sort(key=lambda x: len(adj[x] - visited))
                
                # Randomize slightly
                if rng.random() < 0.3 and len(neighbors) > 1:
                    # Pick second best sometimes
                    next_node = neighbors[1] if len(neighbors) > 1 else neighbors[0]
                else:
                    next_node = neighbors[0]
                
                path.append(next_node)
                visited.add(next_node)
                current = next_node
                
                if len(path) > max_len:
                    break
        
        return best_cycle

    def greedy_decompose(edges, rng):
        """
        Greedily decompose edges into cycles.
        Returns (cycles, remaining_edges)
        """
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cycles = []
        remaining = set(edges)
        
        # Repeat until no more cycles found
        while True:
            # Find a vertex with degree >= 2
            candidates = [u for u in adj if len(adj[u]) >= 2]
            if not candidates:
                break
            
            # Pick a random candidate
            start = rng.choice(candidates)
            
            # Try to find a cycle
            cycle = find_cycle(adj, start, max_len=40, rng=rng)
            
            if cycle:
                cycles.append(cycle)
                # Remove edges from cycle
                for i in range(len(cycle) - 1):
                    u, v = cycle[i], cycle[i+1]
                    e = (min(u, v), max(u, v))
                    remaining.discard(e)
                    adj[u].discard(v)
                    adj[v].discard(u)
            else:
                # No cycle found from this start. 
                # To avoid infinite loop, we can just break or try another start.
                # If we can't find any cycle from any high-degree vertex, we stop.
                # For simplicity, break.
                break
        
        return cycles, remaining

    def local_search_improve(cycles, remaining, edges, rng, time_limit):
        """
        Try to improve the decomposition by merging cycles or finding longer cycles.
        """
        current_cycles = [list(c) for c in cycles]
        current_remaining = set(remaining)
        current_score = len(current_cycles) + len(current_remaining)
        best_score = current_score
        best_cycles = [list(c) for c in current_cycles]
        best_remaining = set(current_remaining)
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Rebuild adj for remaining edges only? No, we need to find cycles in the union of removed cycles.
        # Actually, let's just work with the current state.
        
        T = 5.0
        T_min = 0.1
        cooling = 0.95
        
        while time.time() < time_limit:
            # Pick a random cycle to "break" and re-decompose its edges
            if not current_cycles:
                break
                
            idx = rng.randrange(len(current_cycles))
            cycle_to_break = current_cycles[idx]
            
            # Get edges of this cycle
            cycle_edges = set()
            for i in range(len(cycle_to_break) - 1):
                u, v = cycle_to_break[i], cycle_to_break[i+1]
                cycle_edges.add((min(u, v), max(u, v)))
            
            # Remove this cycle from current state
            new_cycles = [c for i, c in enumerate(current_cycles) if i != idx]
            
            # Pool = cycle_edges + current_remaining? 
            # No, we only re-decompose the edges of the broken cycle.
            # The remaining edges stay as is.
            # But maybe we can find a longer cycle that uses edges from this cycle AND some remaining edges?
            # That would be a "merge".
            
            # Let's try to find a cycle in the subgraph induced by cycle_edges.
            sub_adj = defaultdict(set)
            for u, v in cycle_edges:
                sub_adj[u].add(v)
                sub_adj[v].add(u)
                
            new_sub_cycles = []
            new_sub_remaining = set(cycle_edges)
            
            # Greedily find cycles in this small subgraph
            while True:
                candidates = [u for u in sub_adj if len(sub_adj[u]) >= 2]
                if not candidates:
                    break
                start = rng.choice(candidates)
                c = find_cycle(sub_adj, start, max_len=20, rng=rng)
                if c:
                    new_sub_cycles.append(c)
                    for i in range(len(c) - 1):
                        u, v = c[i], c[i+1]
                        e = (min(u, v), max(u, v))
                        new_sub_remaining.discard(e)
                        sub_adj[u].discard(v)
                        sub_adj[v].discard(u)
                else:
                    break
            
            candidate_cycles = new_cycles + new_sub_cycles
            candidate_remaining = current_remaining | new_sub_remaining
            candidate_score = len(candidate_cycles) + len(candidate_remaining)
            
            delta = candidate_score - current_score
            
            if delta < 0 or (T > T_min and rng.random() < pow(2.71828, -delta/T)):
                current_cycles = candidate_cycles
                current_remaining = candidate_remaining
                current_score = candidate_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_cycles = [list(c) for c in current_cycles]
                    best_remaining = set(current_remaining)
            
            T *= cooling
            
        return best_cycles, best_remaining

    def decompose(edges):
        if not edges:
            return []
        
        rng = random.Random(12345)
        
        # 1. Initial greedy
        cycles, remaining = greedy_decompose(edges, rng)
        
        # 2. Local search
        cycles, remaining = local_search_improve(cycles, remaining, edges, rng, time.time() + 30)
        
        # Convert to pieces
        pieces = []
        for c in cycles:
            pieces.append(c)
        for e in remaining:
            pieces.append([e[0], e[1]])
            
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > DEADLINE:
            # Emergency: return trivial decomposition (all edges)
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
            
        edges = instance(*spec)
        results.append(decompose(edges))
        
    return results
