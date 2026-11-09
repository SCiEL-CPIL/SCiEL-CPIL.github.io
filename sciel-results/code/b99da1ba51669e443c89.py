import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin from 150s limit
    
    n = 40
    target_edges = 401
    
    def build_graph(A_size, seed=None):
        """Build graph with A of given size, B of n-A_size, and internal B edges."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        
        A = list(range(A_size))
        B = list(range(A_size, n))
        
        edges = set()
        
        # All edges between A and B
        for a in A:
            for b in B:
                edges.add((a, b))
                
        needed = target_edges - len(edges)
        
        if needed <= 0:
            return edges, A, B
        
        # Split B into two parts for internal edges
        B_size = n - A_size
        B1_size = B_size // 2
        B1 = B[:B1_size]
        B2 = B[B1_size:]
        
        # Generate all possible pairs between B1 and B2
        pairs = [(u, v) for u in B1 for v in B2]
        rng.shuffle(pairs)
        
        # Also generate pairs within B1 and within B2
        pairs_b1 = [(u, v) for i, u in enumerate(B1) for v in B1[i+1:]]
        pairs_b2 = [(u, v) for i, u in enumerate(B2) for v in B2[i+1:]]
        rng.shuffle(pairs_b1)
        rng.shuffle(pairs_b2)
        
        # Mix all candidate pairs
        all_pairs = pairs + pairs_b1 + pairs_b2
        rng.shuffle(all_pairs)
        
        added = 0
        for u, v in all_pairs:
            if added == needed:
                break
            if (u, v) not in edges:
                edges.add((u, v))
                added += 1
                
        return edges, A, B

    def compute_adj_masks(edges, n):
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        return adj_masks

    def get_max_book_info(edges, adj_masks):
        max_b = -1
        max_edges = []
        for u, v in edges:
            b = (adj_masks[u] & adj_masks[v]).bit_count()
            if b > max_b:
                max_b = b
                max_edges = [(u, v)]
            elif b == max_b:
                max_edges.append((u, v))
        return max_b, max_edges

    def get_sum_sq_books(edges, adj_masks):
        """Sum of squares of book sizes - smooth surrogate for min-max."""
        total = 0
        for u, v in edges:
            b = (adj_masks[u] & adj_masks[v]).bit_count()
            total += b * b
        return total

    # Try multiple initializations and pick the best
    best_edges = None
    best_adj = None
    best_max_book = float('inf')
    best_sum_sq = float('inf')
    
    # Try different A sizes: 8, 9, 10, 11
    for a_size in [8, 9, 10, 11]:
        for seed in range(5):
            edges, A, B = build_graph(a_size, seed)
            if len(edges) != target_edges:
                continue
            adj = compute_adj_masks(edges, n)
            max_b, _ = get_max_book_info(edges, adj)
            sum_sq = get_sum_sq_books(edges, adj)
            
            if max_b < best_max_book or (max_b == best_max_book and sum_sq < best_sum_sq):
                best_max_book = max_b
                best_sum_sq = sum_sq
                best_edges = set(edges)
                best_adj = adj
    
    if best_edges is None:
        # Fallback
        best_edges, _, _ = build_graph(9, 0)
        best_adj = compute_adj_masks(best_edges, n)
        best_max_book, _ = get_max_book_info(best_edges, best_adj)
        best_sum_sq = get_sum_sq_books(best_edges, best_adj)
    
    current_edges = set(best_edges)
    adj_masks = best_adj
    current_max_book = best_max_book
    current_sum_sq = best_sum_sq
    
    def update_adj(u, v, add):
        if add:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        else:
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
    
    # Precompute edge list for faster iteration
    edge_list = list(current_edges)
    
    def get_max_book_info_fast():
        max_b = -1
        max_edges = []
        for u, v in current_edges:
            b = (adj_masks[u] & adj_masks[v]).bit_count()
            if b > max_b:
                max_b = b
                max_edges = [(u, v)]
            elif b == max_b:
                max_edges.append((u, v))
        return max_b, max_edges
    
    def get_sum_sq_fast():
        total = 0
        for u, v in current_edges:
            b = (adj_masks[u] & adj_masks[v]).bit_count()
            total += b * b
        return total
    
    # Local search
    no_improve_count = 0
    max_no_improve = 500
    perturbation_count = 0
    max_perturbations = 20
    
    while time.time() < deadline:
        if current_max_book <= 6:
            break
        
        # If stuck too long, do a perturbation
        if no_improve_count > max_no_improve and perturbation_count < max_perturbations:
            # Do a random perturbation: swap 3-5 random edges
            num_swaps = random.randint(3, 5)
            for _ in range(num_swaps):
                # Pick a random edge to remove
                if not current_edges:
                    break
                u, v = random.choice(list(current_edges))
                current_edges.remove((u, v))
                update_adj(u, v, False)
                
                # Find a random non-edge to add
                found = False
                for _ in range(100):
                    x = random.randint(0, n - 1)
                    y = random.randint(0, n - 1)
                    if x == y:
                        continue
                    if (1 << y) & adj_masks[x]:
                        continue
                    if (x, y) in current_edges or (y, x) in current_edges:
                        continue
                    current_edges.add((x, y))
                    update_adj(x, y, True)
                    found = True
                    break
                
                if not found:
                    # Rollback
                    current_edges.add((u, v))
                    update_adj(u, v, True)
                    break
            
            current_max_book, _ = get_max_book_info_fast()
            current_sum_sq = get_sum_sq_fast()
            no_improve_count = 0
            perturbation_count += 1
            continue
        
        # Pick an edge to remove: prioritize edges in many triangles
        # Get all edges with their book sizes
        edge_books = []
        for u, v in current_edges:
            b = (adj_masks[u] & adj_masks[v]).bit_count()
            edge_books.append((b, u, v))
        
        # Sort by book size descending
        edge_books.sort(reverse=True)
        
        # Pick from top 10% or random if we're not improving
        if no_improve_count < 100:
            # Be more aggressive: pick from top edges
            top_k = max(1, len(edge_books) // 10)
            candidates = edge_books[:top_k]
        else:
            # More exploration: pick from top 50%
            top_k = max(1, len(edge_books) // 2)
            candidates = edge_books[:top_k]
        
        b_remove, u, v = random.choice(candidates)
        
        # Find the best non-edge to add
        # We want to minimize the book size of the new edge
        min_book_new = float('inf')
        best_x, best_y = -1, -1
        best_sum_sq_new = float('inf')
        
        # Iterate over all possible non-edges
        for x in range(n):
            for y in range(x + 1, n):
                # Check if (x,y) is an edge
                if (1 << y) & adj_masks[x]:
                    continue
                if (x, y) in current_edges or (y, x) in current_edges:
                    continue
                
                # Calculate book size for potential new edge
                b = (adj_masks[x] & adj_masks[y]).bit_count()
                
                # Estimate change in sum of squares
                # Removing (u,v) removes b_remove^2 from sum
                # Adding (x,y) adds b^2 to sum
                # But also affects other edges' book sizes
                # For now, use simple heuristic: prefer low b
                if b < min_book_new:
                    min_book_new = b
                    best_x, best_y = x, y
                    if b == 0:
                        break
            if min_book_new == 0:
                break
        
        if best_x == -1:
            break
        
        # Perform the swap: remove (u,v), add (best_x, best_y)
        old_max = current_max_book
        old_sum_sq = current_sum_sq
        
        # Remove (u,v)
        current_edges.remove((u, v))
        update_adj(u, v, False)
        
        # Add (best_x, best_y)
        current_edges.add((best_x, best_y))
        update_adj(best_x, best_y, True)
        
        # Recalculate
        new_max, _ = get_max_book_info_fast()
        new_sum_sq = get_sum_sq_fast()
        
        # Acceptance Criteria
        accepted = False
        if new_max < old_max:
            accepted = True
            no_improve_count = 0
        elif new_max == old_max:
            # Prefer moves that reduce sum of squares
            if new_sum_sq < old_sum_sq:
                accepted = True
                no_improve_count = 0
            elif new_sum_sq == old_sum_sq:
                # Small probability to accept to escape local minima
                if random.random() < 0.05:
                    accepted = True
                else:
                    accepted = False
            else:
                # Only accept if we're stuck
                if no_improve_count > 200 and random.random() < 0.1:
                    accepted = True
                else:
                    accepted = False
        else:
            # Only accept if we're very stuck
            if no_improve_count > 400 and new_max - old_max <= 1 and random.random() < 0.05:
                accepted = True
            else:
                accepted = False

        if accepted:
            current_max_book = new_max
            current_sum_sq = new_sum_sq
        else:
            # Rollback
            current_edges.add((u, v))
            update_adj(u, v, True)
            
            current_edges.remove((best_x, best_y))
            update_adj(best_x, best_y, False)
            
            no_improve_count += 1
            if no_improve_count > max_no_improve + max_perturbations * 100:
                break

    result = [[u, v] for u, v in current_edges]
    return result
