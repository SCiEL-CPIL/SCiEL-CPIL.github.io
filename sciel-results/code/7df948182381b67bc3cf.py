"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

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
    deadline = time.time() + 140.0
    trees = instance_trees(20, 611)
    
    def solve_tree_csp(tree_edges, k, used_edges, deadline, rng):
        """
        CSP-based solver for embedding a tree into K20.
        Uses MRV (Minimum Remaining Values) heuristic with backtracking.
        """
        if k == 1:
            return [0]
        if k == 2:
            for u in range(20):
                for v in range(u + 1, 20):
                    if (u, v) not in used_edges:
                        return [u, v]
            return None

        # Build tree adjacency
        tree_adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
        
        # Root the tree at 0
        parent = [-1] * k
        visited = [False] * k
        visited[0] = True
        queue = [0]
        while queue:
            u = queue.pop(0)
            for v in tree_adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)
        
        # Children list
        children = [[] for _ in range(k)]
        for v in range(1, k):
            children[parent[v]].append(v)
        
        host = [None] * k
        used_hosts = set()
        
        def get_candidates(v, hp):
            """Get valid host candidates for tree vertex v given parent's host hp."""
            candidates = []
            for h in range(20):
                if h in used_hosts:
                    continue
                if hp is not None:
                    edge = (min(hp, h), max(hp, h))
                    if edge in used_edges:
                        continue
                candidates.append(h)
            return candidates
        
        def backtrack(v):
            """Recursively assign host to tree vertex v and its subtree."""
            if time.time() > deadline:
                return False
            
            p = parent[v]
            hp = host[p] if p != -1 else None
            
            candidates = get_candidates(v, hp)
            if not candidates:
                return False
            
            # Heuristic: Order candidates by how many options they leave for children
            # Try candidates that are more "central" first (more available neighbors)
            if len(candidates) > 1:
                def candidate_score(h):
                    # Count how many unused edges from h to other unused hosts
                    score = 0
                    for h2 in range(20):
                        if h2 == h or h2 in used_hosts:
                            continue
                        if (min(h, h2), max(h, h2)) not in used_edges:
                            score += 1
                    return score
                candidates.sort(key=candidate_score, reverse=True)
            
            for h in candidates:
                host[v] = h
                used_hosts.add(h)
                
                # Recurse on children
                success = True
                for child in children[v]:
                    if not backtrack(child):
                        success = False
                        break
                
                if success:
                    return True
                
                # Backtrack
                host[v] = None
                used_hosts.remove(h)
                for child in children[v]:
                    host[child] = None
                    # Also need to clean up descendants - but they should already be None
                    # from failed recursion, so we just need to remove from used_hosts
                    # Actually, we need to properly clean up. Let's use a different approach.
            
            return False
        
        # The above backtracking has a cleanup issue. Let's use a cleaner approach
        # with explicit state tracking.
        
        host = [None] * k
        used_hosts = set()
        
        def backtrack_clean(v):
            if time.time() > deadline:
                return False
            
            p = parent[v]
            hp = host[p] if p != -1 else None
            
            candidates = get_candidates(v, hp)
            if not candidates:
                return False
            
            # Order by centrality heuristic
            if len(candidates) > 1:
                def candidate_score(h):
                    score = 0
                    for h2 in range(20):
                        if h2 == h or h2 in used_hosts:
                            continue
                        if (min(h, h2), max(h, h2)) not in used_edges:
                            score += 1
                    return score
                candidates.sort(key=candidate_score, reverse=True)
            
            for h in candidates:
                host[v] = h
                used_hosts.add(h)
                
                success = True
                for child in children[v]:
                    if not backtrack_clean(child):
                        success = False
                        break
                
                if success:
                    return True
                
                # Clean up: remove h and all descendants' assignments
                used_hosts.remove(h)
                host[v] = None
                for child in children[v]:
                    # Clean up subtree rooted at child
                    def clean_subtree(c):
                        for gc in children[c]:
                            clean_subtree(gc)
                        if host[c] is not None:
                            used_hosts.remove(host[c])
                            host[c] = None
                    clean_subtree(child)
            
            return False
        
        if backtrack_clean(0):
            return host
        return None

    best_score = 0
    best_embeddings = [None] * 20
    best_embeddings[0] = [0]
    
    # Try multiple strategies with different random seeds
    for restart in range(100):
        if time.time() > deadline - 2:
            break
            
        rng = Random(1234 + restart)
        used_edges = set()
        embeddings = [None] * 20
        embeddings[0] = [0]
        
        # Strategy: Try different orders
        # Order 1: T20 first, then T19 down to T2
        # Order 2: T2 first, then T3 up to T20
        # Order 3: Alternating large and small
        
        if restart % 3 == 0:
            # T20 first
            order = list(range(19, 1, -1))  # T20, T19, ..., T2
        elif restart % 3 == 1:
            # T2 first
            order = list(range(1, 20))  # T2, T3, ..., T20
        else:
            # Mixed: large, small, large, small...
            order = []
            for i in range(10):
                if 19 - i >= 1:
                    order.append(19 - i)
                if i + 1 <= 19:
                    order.append(i + 1)
            order = [x for x in order if 1 <= x <= 19]
        
        success = True
        for k_idx in order:
            k = k_idx + 1  # Convert to 1-indexed tree size
            if time.time() > deadline - 2:
                success = False
                break
            
            res = solve_tree_csp(trees[k-1], k, used_edges, deadline, rng)
            if res is None:
                success = False
                break
            
            embeddings[k-1] = res
            for u, v in trees[k-1]:
                used_edges.add((min(res[u], res[v]), max(res[u], res[v])))
                
        if success:
            score = 0
            for k in range(1, 21):
                if embeddings[k-1] is not None:
                    score += k - 1
            if score > best_score:
                best_score = score
                best_embeddings = embeddings[:]
                
            if best_score == 190:
                break

    # Fallback: If we didn't get a good solution, try to fill in missing trees greedily
    if best_score < 190:
        # Collect used edges from existing embeddings
        used_edges = set()
        for k in range(2, 21):
            if best_embeddings[k-1] is not None:
                for u, v in trees[k-1]:
                    used_edges.add((min(best_embeddings[k-1][u], best_embeddings[k-1][v]), 
                                    max(best_embeddings[k-1][u], best_embeddings[k-1][v])))
        
        # Try to embed missing trees with a simpler greedy approach
        for k in range(2, 21):
            if best_embeddings[k-1] is not None:
                continue
            if time.time() > deadline - 1:
                break
            
            # Try random embeddings
            for attempt in range(50):
                if time.time() > deadline - 1:
                    break
                rng2 = Random(9999 + k * 100 + attempt)
                embedding = list(range(k))
                rng2.shuffle(embedding)
                
                # Check if valid
                valid = True
                for u, v in trees[k-1]:
                    edge = (min(embedding[u], embedding[v]), max(embedding[u], embedding[v]))
                    if edge in used_edges:
                        valid = False
                        break
                
                if valid:
                    best_embeddings[k-1] = embedding
                    for u, v in trees[k-1]:
                        used_edges.add((min(embedding[u], embedding[v]), max(embedding[u], embedding[v])))
                    best_score += k - 1
                    break
    
    # Final validation
    for i in range(20):
        if best_embeddings[i] is not None:
            if len(best_embeddings[i]) != i + 1:
                best_embeddings[i] = None
            else:
                if len(set(best_embeddings[i])) != i + 1:
                    best_embeddings[i] = None
                else:
                    # Ensure all are Python ints
                    best_embeddings[i] = [int(x) for x in best_embeddings[i]]
    
    return best_embeddings

# EVOLVE-BLOCK-END
