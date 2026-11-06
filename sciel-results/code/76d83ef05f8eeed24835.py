import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 150.0 # seconds
    
    def random_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj, R, P, X, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Pivot selection: choose v in P union X with max degree in P
        P_union_X = P.union(X)
        if P_union_X:
            v = max(P_union_X, key=lambda x: len(adj[x].intersection(P)))
        else:
            return
            
        # Candidates are P \ N(v)
        candidates = P.difference(adj[v])
        
        for u in list(candidates):
            bron_kerbosch(adj, R.union({u}), P.intersection(adj[u]), X.intersection(adj[u]), cliques)
            P.remove(u)
            X.add(u)

    def solve_set_cover(cliques, n):
        if not cliques:
            return []
        
        # Map clique to index
        # Represent set of covered cliques as a bitmask (int)
        # If number of cliques > 60, we might need a different representation, 
        # but for n=40, p=0.2, max cliques is likely < 100.
        # Python ints are arbitrary precision, so this is fine.
        
        m = len(cliques)
        # Precompute vertex to clique mask
        v_cliques = [0] * n
        for i, c in enumerate(cliques):
            mask = 1 << i
            for v in c:
                v_cliques[v] |= mask
        
        all_cliques_mask = (1 << m) - 1
        
        best = {'size': n, 'set': set()}
        
        # Sort vertices by frequency of appearance in cliques (descending)
        # This helps in finding a good initial solution
        v_order = sorted(range(n), key=lambda v: bin(v_cliques[v]).count('1'), reverse=True)
        
        # Greedy initial solution
        covered = 0
        greedy_set = set()
        remaining_cliques = set(range(m))
        # Simple greedy: pick vertex covering most uncovered cliques
        while remaining_cliques:
            best_v = -1
            best_count = -1
            for v in v_order:
                if v in greedy_set: continue
                count = 0
                # Count how many remaining cliques v covers
                # This is expensive if done naively. 
                # We can precompute or just iterate.
                # Since m is small, let's just iterate.
                for i in remaining_cliques:
                    if v_cliques[v] & (1 << i):
                        count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v == -1:
                break
            greedy_set.add(best_v)
            # Update remaining
            new_remaining = set()
            for i in remaining_cliques:
                if not (v_cliques[best_v] & (1 << i)):
                    new_remaining.add(i)
            remaining_cliques = new_remaining
            if len(greedy_set) >= best['size']:
                break
        if len(greedy_set) < best['size']:
            best['size'] = len(greedy_set)
            best['set'] = set(greedy_set)
            
        # Branch and Bound
        def branch(covered_mask, current_size, current_set, depth):
            if time.time() > deadline - 5:
                return
            
            if current_size >= best['size']:
                return
                
            if covered_mask == all_cliques_mask:
                if current_size < best['size']:
                    best['size'] = current_size
                    best['set'] = set(current_set)
                return
                
            # Lower bound: 
            # Find an uncovered clique
            uncovered_cliques = [i for i in range(m) if not (covered_mask & (1 << i))]
            if not uncovered_cliques:
                return
                
            # Heuristic: Pick a clique with fewest available vertices? 
            # Or just pick the first one.
            # To improve pruning, pick the "hardest" clique.
            # Let's pick the first uncovered clique for simplicity, or one with min degree.
            # Let's try picking the clique with the minimum number of vertices not in current_set?
            # Actually, we just need to branch on vertices of an uncovered clique.
            
            # Select a representative uncovered clique
            # We can sort uncovered cliques by size or something.
            # Let's just take the first one.
            target_clique_idx = uncovered_cliques[0]
            target_clique = cliques[target_clique_idx]
            
            # Branch on each vertex in this clique
            # Order vertices by how many other uncovered cliques they cover
            candidates = []
            for v in target_clique:
                if v in current_set:
                    continue
                # Count new coverage
                new_cover = v_cliques[v] & (~covered_mask)
                candidates.append((bin(new_cover).count('1'), v))
            
            # Sort by coverage descending
            candidates.sort(reverse=True)
            
            for _, v in candidates:
                new_covered = covered_mask | v_cliques[v]
                branch(new_covered, current_size + 1, current_set | {v}, depth + 1)
                if time.time() > deadline - 5:
                    return

        branch(0, 0, set(), 0)
        
        return sorted(list(best['set']))

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 10:
            # Fallback: return all vertices to ensure validity
            results.append(list(range(n)))
            continue
            
        adj = random_graph(n, p, seed)
        cliques = []
        # Bron-Kerbosch
        # R: current clique
        # P: candidates that can extend R
        # X: candidates that can extend R but were already processed
        bron_kerbosch(adj, set(), set(range(n)), set(), cliques)
        
        # Filter cliques size >= 2
        cliques = [c for c in cliques if len(c) >= 2]
        
        solution = solve_set_cover(cliques, n)
        results.append(solution)
        
    return results

# Test locally
if __name__ == "__main__":
    res = construct()
    print(res)
