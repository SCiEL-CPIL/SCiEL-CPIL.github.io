"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from random import Random
import random

def _instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    trees = _instance_trees(20, 611)
    n = 20
    rng = random.Random(777)
    
    # Local search with restarts
    best_result = None
    best_cost = float('inf')
    
    for restart in range(50):
        # Initialize with random embeddings
        embeddings = [None] * 21
        for k in range(1, 21):
            if k == 1:
                embeddings[k] = [0]
            else:
                hosts = list(range(n))
                rng.shuffle(hosts)
                embeddings[k] = hosts[:k]
        
        def compute_cost(embeddings, trees, n):
            edge_to_trees = {}
            for k in range(1, 21):
                tree_edges = trees[k - 1]
                emb = embeddings[k]
                for u, v in tree_edges:
                    hu, hv = emb[u], emb[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge not in edge_to_trees:
                        edge_to_trees[edge] = []
                    edge_to_trees[edge].append(k)
            
            cost = 0
            for edge, tree_list in edge_to_trees.items():
                if len(tree_list) > 1:
                    cost += len(tree_list) - 1
            return cost
        
        current_cost = compute_cost(embeddings, trees, n)
        
        # Local search
        for iteration in range(5000):
            if current_cost == 0:
                break
            
            # Pick a random tree involved in a conflict, or random tree
            # For simplicity, pick random tree
            k = rng.randint(1, 20)
            
            old_emb = embeddings[k][:]
            best_new_cost = current_cost
            best_new_emb = old_emb
            
            # Try several random re-embeddings
            for attempt in range(50):
                new_emb = old_emb[:]
                # Randomly permute the embedding
                hosts = list(range(n))
                rng.shuffle(hosts)
                new_emb = hosts[:k]
                
                embeddings[k] = new_emb
                new_cost = compute_cost(embeddings, trees, n)
                
                if new_cost < best_new_cost:
                    best_new_cost = new_cost
                    best_new_emb = new_emb[:]
            
            embeddings[k] = best_new_emb
            current_cost = best_new_cost
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_result = [embeddings[k] for k in range(1, 21)]
        
        if best_cost == 0:
            break
    
    if best_result is None:
        # Fallback: return identity embeddings (will likely be invalid)
        best_result = [list(range(k)) for k in range(1, 21)]
    
    return best_result
# EVOLVE-BLOCK-END
