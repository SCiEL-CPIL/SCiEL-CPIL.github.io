import time
import random

def construct():
    n = 40
    m = 401
    start_time = time.time()
    deadline = start_time + 115  # 5s margin before 120s limit

    def build_masks(edges):
        masks = [0] * n
        for u, v in edges:
            masks[u] |= 1 << v
            masks[v] |= 1 << u
        return masks

    def compute_books(masks, edges):
        books = {}
        max_b = 0
        for u, v in edges:
            b = (masks[u] & masks[v]).bit_count()
            books[(u, v)] = b
            if b > max_b:
                max_b = b
        return books, max_b

    def max_book(masks, edges):
        _, max_b = compute_books(masks, edges)
        return max_b

    def bipartite_base():
        """Start from K(20,20) plus one intra-partition edge."""
        part_a = list(range(20))
        part_b = list(range(20, 40))
        edges = set()
        for a in part_a:
            for b in part_b:
                edges.add((a, b))
        # Add one intra-edge within part_a
        edges.add((0, 1))
        return edges

    def targeted_local_search(edges, masks, deadline):
        """Focus swaps on edges with high book sizes."""
        books, current_max = compute_books(masks, edges)
        best_score = current_max
        best_edges = set(edges)
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        edge_set = set(edges)
        non_edges = [p for p in all_pairs if p not in edge_set]
        
        no_improve = 0
        step = 0
        
        while time.time() < deadline:
            # Find edges with book size >= current_max
            hot_edges = [(e, b) for e, b in books.items() if b >= current_max]
            if not hot_edges:
                hot_edges = list(books.items())
            
            # Pick a hot edge to remove
            (u, v), _ = random.choice(hot_edges)
            
            # Pick a non-edge to add - prefer non-edges that connect
            # vertices with low common neighbors
            candidates = non_edges
            if len(candidates) > 50:
                # Sample a subset for speed
                candidates = random.sample(candidates, 50)
            
            # Score each candidate by estimated impact
            best_candidate = None
            best_candidate_score = None
            for x, y in candidates:
                # Estimate: adding (x,y) creates triangles with common neighbors
                common_xy = (masks[x] & masks[y]).bit_count()
                # Removing (u,v) destroys books[u,v] triangles
                # Net change depends on which triangles overlap
                # Heuristic: prefer candidates with low common_xy
                score = common_xy
                if best_candidate_score is None or score < best_candidate_score:
                    best_candidate_score = score
                    best_candidate = (x, y)
            
            x, y = best_candidate
            
            # Perform swap
            edges.discard((u, v))
            edges.add((x, y))
            
            # Update masks
            masks[u] &= ~(1 << v)
            masks[v] &= ~(1 << u)
            masks[x] |= 1 << y
            masks[y] |= 1 << x
            
            # Update non_edges list
            non_edges.remove((x, y))
            non_edges.append((u, v))
            
            # Recompute books
            books, new_score = compute_books(masks, edges)
            
            if new_score <= current_max:
                current_max = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_edges = set(edges)
                    no_improve = 0
                else:
                    if random.random() < 0.2:
                        no_improve += 1
                    else:
                        # Revert
                        edges.discard((x, y))
                        edges.add((u, v))
                        masks[x] &= ~(1 << y)
                        masks[y] &= ~(1 << x)
                        masks[u] |= 1 << v
                        masks[v] |= 1 << u
                        non_edges.remove((u, v))
                        non_edges.append((x, y))
                        books, current_max = compute_books(masks, edges)
                        no_improve += 1
            else:
                # Revert
                edges.discard((x, y))
                edges.add((u, v))
                masks[x] &= ~(1 << y)
                masks[y] &= ~(1 << x)
                masks[u] |= 1 << v
                masks[v] |= 1 << u
                non_edges.remove((u, v))
                non_edges.append((x, y))
                books, current_max = compute_books(masks, edges)
                no_improve += 1
            
            step += 1
            
            # If stuck for too long, try a more aggressive restart
            if no_improve > 3000:
                # Perturb: do 10 random swaps
                for _ in range(10):
                    u2, v2 = random.choice(list(edges))
                    x2, y2 = random.choice(non_edges)
                    edges.discard((u2, v2))
                    edges.add((x2, y2))
                    masks[u2] &= ~(1 << v2)
                    masks[v2] &= ~(1 << u2)
                    masks[x2] |= 1 << y2
                    masks[y2] |= 1 << x2
                    non_edges.remove((x2, y2))
                    non_edges.append((u2, v2))
                books, current_max = compute_books(masks, edges)
                no_improve = 0
        
        return best_edges, best_score

    # Phase 1: Start from bipartite base
    edges = bipartite_base()
    masks = build_masks(edges)
    books, current_max = compute_books(masks, edges)
    
    best_score = current_max
    best_edges = set(edges)
    
    # Phase 2: Targeted local search
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        edges, score = targeted_local_search(edges, masks, deadline)
        if score < best_score:
            best_score = score
            best_edges = edges
    
    # Phase 3: Try multiple random restarts if time remains
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        for restart in range(5):
            if time.time() >= deadline - 5:
                break
            # Random graph initialization
            rand_edges = set()
            while len(rand_edges) < m:
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    rand_edges.add((min(u, v), max(u, v)))
            rand_masks = build_masks(rand_edges)
            rand_books, rand_max = compute_books(rand_masks, rand_edges)
            
            # Short local search on random start
            remaining = deadline - time.time()
            if remaining > 5:
                rand_edges, rand_score = targeted_local_search(rand_edges, rand_masks, deadline)
                if rand_score < best_score:
                    best_score = rand_score
                    best_edges = rand_edges
    
    return [[u, v] for u, v in best_edges]
