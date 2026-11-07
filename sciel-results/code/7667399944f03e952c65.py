"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random, sample
from itertools import combinations

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    trees = instance_trees(20, 611)
    n = 20
    
    # All edges in K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    # Order trees by difficulty (most edges first for constraint propagation)
    tree_order = list(range(20))  # indices 0..19 for T1..T20
    tree_order.sort(key=lambda idx: (-(len(trees[idx])), -idx))
    
    def embed_tree(tree_idx, available_edges, rng, max_nodes=20):
        """Try to embed tree tree_idx into available_edges using backtracking with pruning.
        Returns a list of k distinct int host vertices (position i = host for tree vertex i),
        or None if no embedding found."""
        tree_edges = trees[tree_idx]
        k = tree_idx + 1
        if k == 1:
            # T1: no edges, pick any vertex
            return [rng.randint(0, n - 1)]
        
        # Build tree adjacency
        tree_adj = {v: [] for v in range(k)}
        for u, v in tree_edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
        
        # Order tree vertices: highest degree first for better pruning
        ordered_tv = sorted(range(k), key=lambda v: (-len(tree_adj[v]), v))
        
        host_map = {}
        used_hosts = set()
        
        # Precompute: for each tree vertex, which host vertices have enough available
        # incident edges to connect to its tree neighbors
        def get_candidates(tv, assigned):
            """Get candidate host vertices for tree vertex tv given current assignments."""
            candidates = []
            for h in range(n):
                if h in used_hosts:
                    continue
                # Check if h has available edges to all assigned neighbors
                ok = True
                for nb in tree_adj[tv]:
                    if nb in assigned:
                        hn = assigned[nb]
                        edge = (min(h, hn), max(h, hn))
                        if edge not in available_edges:
                            ok = False
                            break
                if ok:
                    candidates.append(h)
            return candidates
        
        # Randomize within same-degree groups for diversity
        def randomized_order():
            deg_groups = {}
            for v in ordered_tv:
                d = len(tree_adj[v])
                if d not in deg_groups:
                    deg_groups[d] = []
                deg_groups[d].append(v)
            result = []
            for d in sorted(deg_groups.keys(), reverse=True):
                group = deg_groups[d][:]
                rng.shuffle(group)
                result.extend(group)
            return result
        
        ro = randomized_order()
        
        def backtrack(pos):
            if pos == k:
                # Verify all edges
                for u, v in tree_edges:
                    hu, hv = host_map[u], host_map[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge not in available_edges:
                        return False
                return True
            
            tv = ro[pos]
            cands = get_candidates(tv, host_map)
            if not cands:
                return False
            
            # Sort candidates by how many available edges they have to unassigned neighbors
            # (fewer options first for stronger pruning)
            def cand_score(h):
                score = 0
                for nb in tree_adj[tv]:
                    if nb not in host_map:
                        edge = (min(h, nb), max(h, nb))  # placeholder, not meaningful
                        # Count available edges from h to potential hosts of nb
                        # This is expensive; use simpler heuristic
                        pass
                # Simpler: prefer hosts with fewer remaining available edges (more constrained)
                avail_count = 0
                for nb_h in range(n):
                    if nb_h != h and nb_h not in used_hosts:
                        edge = (min(h, nb_h), max(h, nb_h))
                        if edge in available_edges:
                            avail_count += 1
                return avail_count
            
            cands.sort(key=cand_score)
            
            for h in cands:
                host_map[tv] = h
                used_hosts.add(h)
                if backtrack(pos + 1):
                    return True
                del host_map[tv]
                used_hosts.remove(h)
            
            return False
        
        if backtrack(0):
            return [host_map[i] for i in range(k)]
        return None
    
    # Main search with randomized restarts
    best_result = None
    total_tree_edges = sum(len(trees[i]) for i in range(20))
    
    for attempt in range(50000):
        if time.time() > deadline:
            break
        
        rng = Random(attempt * 7919 + 13)
        
        available = set(all_edges)
        result = [None] * 20
        success = True
        
        for t_idx in tree_order:
            if time.time() > deadline:
                success = False
                break
            
            k = t_idx + 1
            embedded = None
            
            # Try multiple randomizations per tree
            max_trials = 30 if k <= 10 else 10
            for trial in range(max_trials):
                embedded = embed_tree(t_idx, available, rng)
                if embedded is not None:
                    break
            
            if embedded is None:
                success = False
                break
            
            result[t_idx] = embedded
            # Remove used edges
            for u, v in trees[t_idx]:
                hu, hv = embedded[u], embedded[v]
                edge = (min(hu, hv), max(hu, hv))
                available.discard(edge)
        
        if success:
            # Verify: all 190 edges should be used exactly once
            if len(available) == 0:
                best_result = result
                break
            # Even if not perfect, it's a valid edge-disjoint embedding
            # But we want maximum value, so prefer perfect
            if best_result is None:
                best_result = result
    
    if best_result is None:
        # Fallback: try deterministic approach
        rng = Random(42)
        available = set(all_edges)
        result = [None] * 20
        success = True
        for t_idx in tree_order:
            k = t_idx + 1
            embedded = embed_tree(t_idx, available, rng)
            if embedded is None:
                success = False
                break
            result[t_idx] = embedded
            for u, v in trees[t_idx]:
                hu, hv = embedded[u], embedded[v]
                edge = (min(hu, hv), max(hu, hv))
                available.discard(edge)
        if success:
            best_result = result
    
    if best_result is None:
        return [None] * 19 + [list(range(20))]
    
    # Ensure types are correct: lists of Python ints
    final = []
    for entry in best_result:
        if entry is None:
            final.append(None)
        else:
            final.append([int(x) for x in entry])
    return final

# EVOLVE-BLOCK-END
