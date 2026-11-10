"""Rewritten program: decompose each instance graph into edge-disjoint cycles and edges, minimizing pieces."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    deadline = time.time() + 145  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_2_core(edge_set):
        if not edge_set:
            return set()
        deg = defaultdict(int)
        adj = defaultdict(set)
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
            
        q = deque([v for v in deg if deg[v] < 2])
        in_core = {v: True for v in deg}
        
        while q:
            v = q.popleft()
            if not in_core[v]:
                continue
            in_core[v] = False
            for nb in adj[v]:
                if in_core[nb]:
                    deg[nb] -= 1
                    if deg[nb] < 2:
                        q.append(nb)
                        
        core_edges = set()
        for u, v in edge_set:
            if in_core[u] and in_core[v]:
                core_edges.add((u, v))
        return core_edges

    def find_cycle_in_core(core_edges):
        if not core_edges:
            return None
        adj = defaultdict(set)
        for u, v in core_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        start = random.choice(list(adj.keys()))
        visited = {start}
        stack = [start]
        
        while stack:
            u = stack[-1]
            nbrs = list(adj[u])
            random.shuffle(nbrs)
            
            advanced = False
            for nb in nbrs:
                if nb == start and len(stack) >= 2:
                    return stack[:]
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
                    advanced = True
                    break
                    
            if not advanced:
                stack.pop()
                
        return None

    def greedy_decompose(edge_set):
        """
        Greedily extract cycles from edge_set.
        Returns (list_of_cycles, set_of_remaining_edges)
        """
        rem = set(edge_set)
        cycles = []
        
        while rem:
            core = get_2_core(rem)
            if not core:
                break
                
            cycle = find_cycle_in_core(core)
            if not cycle:
                break
                
            # Remove cycle edges
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                rem.discard(e)
            cycles.append(cycle)
            
        return cycles, rem

    def score_decomposition(cycles):
        """
        Score is sum(len(cycle) - 1). Higher is better.
        """
        return sum(len(c) - 1 for c in cycles)

    def local_search(initial_cycles, initial_remaining, time_limit):
        """
        Perform local search to improve the decomposition.
        """
        current_cycles = [c[:] for c in initial_cycles]
        current_remaining = set(initial_remaining)
        current_score = score_decomposition(current_cycles)
        
        best_cycles = [c[:] for c in current_cycles]
        best_remaining = set(current_remaining)
        best_score = current_score
        
        # Simple hill climbing
        while time.time() < time_limit:
            if not current_cycles:
                break
                
            # Pick a random cycle to remove
            idx = random.randrange(len(current_cycles))
            c_old = current_cycles[idx]
            
            # Restore its edges to the remaining set
            edges_to_add = set()
            for i in range(len(c_old)):
                u, v = c_old[i], c_old[(i + 1) % len(c_old)]
                edges_to_add.add(tuple(sorted((u, v))))
                
            new_remaining = current_remaining | edges_to_add
            
            # Remove the cycle from current list
            current_cycles.pop(idx)
            
            # Re-greedy the new remaining set
            new_cycles, new_remaining_final = greedy_decompose(new_remaining)
            
            new_score = score_decomposition(new_cycles)
            
            # Check if this move improved the score
            # Note: The "remaining" edges are just the forest part, they don't contribute to score.
            # The score is purely from the cycles.
            if new_score > best_score:
                best_score = new_score
                best_cycles = [c[:] for c in new_cycles]
                best_remaining = set(new_remaining_final)
                
                # Update current state to the new one
                current_cycles = best_cycles
                current_remaining = best_remaining
                current_score = new_score
            else:
                # Revert the move (conceptually, we just don't update best)
                # But we need to restore the cycle to current_cycles for next iteration?
                # Actually, we can just keep the current state as is (before the move)
                # So we don't change current_cycles or current_remaining.
                pass
                
        return best_cycles, best_remaining

    def decompose(edges, n, time_budget):
        if not edges:
            return []

        edge_set = set(tuple(sorted(e)) for e in edges)
        
        best_cycles = []
        best_remaining = set(edge_set)
        best_score = 0
        
        # Multiple restarts of (Greedy + Local Search)
        # Estimate time per restart
        # For n=30, greedy is fast. Local search adds overhead.
        # Let's do as many restarts as possible within the budget.
        
        # First, do a few pure greedy restarts to get a good baseline
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            cycles, rem = greedy_decompose(edge_set)
            score = score_decomposition(cycles)
            if score > best_score:
                best_score = score
                best_cycles = [c[:] for c in cycles]
                best_remaining = set(rem)
                
        # Now, apply local search to the best solution found so far
        # And potentially restart local search from other good solutions
        # For simplicity, let's just run local search on the best greedy solution
        # and also on a few other good greedy solutions
        
        # Run local search on the best solution
        if time.time() < deadline - 2:
            ls_cycles, ls_rem = local_search(best_cycles, best_remaining, deadline - 2)
            ls_score = score_decomposition(ls_cycles)
            if ls_score > best_score:
                best_score = ls_score
                best_cycles = ls_cycles
                best_remaining = ls_rem
                
        # Construct the final list of pieces
        pieces = []
        for c in best_cycles:
            pieces.append(c[:])
        for e in best_remaining:
            pieces.append(list(e))
            
        return pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    for spec in specs:
        if time.time() > deadline - 5:
            # Fallback: all edges
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n, deadline - time.time())
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
