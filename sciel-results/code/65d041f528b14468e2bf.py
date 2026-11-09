import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 285  # Safety margin (330s limit, 45s buffer)
    
    N = 64
    M = N * N
    
    # Precompute all points
    points = [(x, y) for x in range(N) for y in range(N)]
    coords = np.array(points, dtype=np.int32)
    
    # Precompute squared distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sum(diff**2, axis=-1)
    
    # Precompute for each point, which other points have the same distance
    # This helps quickly check if adding p creates an isosceles triangle
    # For each pair (i,j), dist[i,j] is the squared distance
    # We need: for any three points a,b,c, dist[a,b] != dist[a,c] != dist[b,c]
    # Equivalently: no point can have two other points at the same distance
    
    # For faster checking: for each point p, maintain a count of how many selected points
    # are at each distance from p. If any count >= 2, then adding p would create isosceles.
    # Also need to check that p doesn't create isosceles with existing pairs.
    
    def try_add(S_set, dist_to_S, p):
        """Check if point p can be added to set S_set."""
        if p in S_set:
            return False
        
        if not S_set:
            return True
        
        # Check 1: All distances from p to S must be unique
        s_indices = list(S_set)
        d_vals = dist[p, s_indices]
        
        # Check for duplicate distances from p
        seen_d = set()
        for d in d_vals:
            d_int = int(d)
            if d_int in seen_d:
                return False
            seen_d.add(d_int)
        
        # Check 2: For each s in S, d(p,s) must not be in dist_to_S[s]
        for i, s in enumerate(s_indices):
            d_int = int(d_vals[i])
            if d_int in dist_to_S[s]:
                return False
        
        return True

    def add_point(S_set, dist_to_S, p):
        """Add point p to S_set and update dist_to_S."""
        S_set.add(p)
        dist_to_S[p] = set()
        for q in S_set:
            if q == p:
                continue
            d = int(dist[p, q])
            dist_to_S[p].add(d)
            dist_to_S[q].add(d)

    def remove_point(S_set, dist_to_S, p):
        """Remove point p from S_set and update dist_to_S."""
        if p not in S_set:
            return
        for q in S_set:
            if q == p:
                continue
            d = int(dist[p, q])
            dist_to_S[q].discard(d)
        dist_to_S.pop(p, None)
        S_set.discard(p)

    def greedy_construct(order):
        """Greedy construction following a given order."""
        S_set = set()
        dist_to_S = {}
        for p in order:
            if time.time() > deadline - 5:
                break
            if try_add(S_set, dist_to_S, p):
                add_point(S_set, dist_to_S, p)
        return S_set, dist_to_S

    def local_search(S_set, dist_to_S, max_iters=2000):
        """Local search with add, 1-swap, 2-swap, and kick operations."""
        S_set = set(S_set)
        dist_to_S = {k: set(v) for k, v in dist_to_S.items()}
        
        # Maintain a list of unselected points for faster candidate generation
        unselected = set(range(M)) - S_set
        
        improved = True
        iter_count = 0
        while improved and time.time() < deadline - 15 and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # 1. Try to add any point
            if unselected:
                candidates = list(unselected)
                random.shuffle(candidates)
                for p in candidates:
                    if try_add(S_set, dist_to_S, p):
                        add_point(S_set, dist_to_S, p)
                        unselected.discard(p)
                        improved = True
                        break
            
            if improved:
                continue
            
            # 2. If no addition, try 1-swap
            if S_set:
                s_list = list(S_set)
                random.shuffle(s_list)
                
                swap_success = False
                # Try removing up to 10 random points
                for r in s_list[:10]:
                    remove_point(S_set, dist_to_S, r)
                    unselected.add(r)
                    
                    candidates = list(unselected)
                    random.shuffle(candidates)
                    
                    for p in candidates:
                        if p == r:
                            continue
                        if try_add(S_set, dist_to_S, p):
                            add_point(S_set, dist_to_S, p)
                            unselected.discard(p)
                            improved = True
                            swap_success = True
                            break
                    
                    if not swap_success:
                        add_point(S_set, dist_to_S, r)
                        unselected.discard(r)
                    else:
                        break
                
                if improved:
                    continue

            # 3. Try 2-swap (remove 2, add 2)
            if len(S_set) >= 2:
                s_list = list(S_set)
                random.shuffle(s_list)
                
                # Try a few pairs of removals
                num_pairs = min(5, len(s_list))
                for i in range(num_pairs):
                    r1 = s_list[i]
                    r2 = s_list[(i + 1) % len(s_list)]
                    if r1 == r2:
                        continue
                    
                    remove_point(S_set, dist_to_S, r1)
                    remove_point(S_set, dist_to_S, r2)
                    unselected.add(r1)
                    unselected.add(r2)
                    
                    cand_list = list(unselected)
                    random.shuffle(cand_list)
                    
                    success = False
                    # Limit search for speed
                    search_limit = min(300, len(cand_list))
                    for p1 in cand_list[:search_limit]:
                        if try_add(S_set, dist_to_S, p1):
                            add_point(S_set, dist_to_S, p1)
                            unselected.discard(p1)
                            for p2 in cand_list[:search_limit]:
                                if p2 == p1 or p2 not in unselected:
                                    continue
                                if try_add(S_set, dist_to_S, p2):
                                    add_point(S_set, dist_to_S, p2)
                                    unselected.discard(p2)
                                    success = True
                                    improved = True
                                    break
                            if not success:
                                remove_point(S_set, dist_to_S, p1)
                                unselected.add(p1)
                            else:
                                break
                    
                    if not success:
                        add_point(S_set, dist_to_S, r1)
                        add_point(S_set, dist_to_S, r2)
                        unselected.discard(r1)
                        unselected.discard(r2)
                    else:
                        break
                
                if improved:
                    continue

            # 4. Kick: Remove 4 random points, re-greedy
            if S_set:
                s_list = list(S_set)
                random.shuffle(s_list)
                kick_points = s_list[:4]
                for r in kick_points:
                    remove_point(S_set, dist_to_S, r)
                    unselected.add(r)
                
                # Re-greedy: Try to add as many points as possible
                cand_list = list(unselected)
                random.shuffle(cand_list)
                
                kick_improved = False
                added_count = 0
                for p in cand_list:
                    if time.time() > deadline - 15:
                        break
                    if p not in unselected:
                        continue
                    if try_add(S_set, dist_to_S, p):
                        add_point(S_set, dist_to_S, p)
                        unselected.discard(p)
                        kick_improved = True
                        added_count += 1
                        # Stop after adding a few to keep it fast
                        if added_count >= 4:
                            break
                        
                if kick_improved:
                    improved = True
                else:
                    for r in kick_points:
                        add_point(S_set, dist_to_S, r)
                        unselected.discard(r)

        return S_set, dist_to_S

    def generate_sidon_order(N, seed_offset=0):
        """Generate an ordering based on a Sidon-like sequence for good initialization."""
        # Use a simple Sidon-like construction: points where x and y are from
        # a quadratic residue sequence or similar
        # For a quick approximation, use points where x^2 + y^2 mod p are distinct
        # Or use a spiral pattern that spreads points out
        
        # Spiral ordering from center
        center = N // 2
        order = []
        for r in range(max(center, N - center)):
            # Points at distance r from center in a spiral
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:  # Only the outer ring
                        x = center + dx
                        y = center + dy
                        if 0 <= x < N and 0 <= y < N:
                            idx = y * N + x
                            if idx not in order:
                                order.append(idx)
        
        # Fill in any missing points
        for i in range(M):
            if i not in order:
                order.append(i)
        
        # Shuffle slightly to add randomness
        for i in range(len(order) - 1, 0, -1):
            j = random.randint(0, i)
            order[i], order[j] = order[j], order[i]
        
        return order

    best_S_set = set()
    best_dist_to_S = {}
    
    # Generate multiple different orderings for restarts
    orders = []
    
    # 1. Pure random orders
    for _ in range(15):
        o = list(range(M))
        random.shuffle(o)
        orders.append(o)
    
    # 2. Spiral/center-out ordering
    orders.append(generate_sidon_order(N))
    
    # 3. Row-major with some perturbation
    o = list(range(M))
    for i in range(len(o) - 1, 0, -1):
        if random.random() < 0.3:  # 30% chance to swap
            j = random.randint(0, i)
            o[i], o[j] = o[j], o[i]
    orders.append(o)
    
    # 4. Column-major ordering
    col_order = []
    for x in range(N):
        for y in range(N):
            col_order.append(y * N + x)
    random.shuffle(col_order)
    orders.append(col_order)
    
    # 5. Diagonal ordering
    diag_order = []
    for d in range(2 * N - 1):
        for x in range(N):
            y = d - x
            if 0 <= y < N:
                diag_order.append(y * N + x)
    random.shuffle(diag_order)
    orders.append(diag_order)
    
    # Run greedy + local search for each order
    for o in orders:
        if time.time() > deadline - 30:
            break
            
        S_set, dist_to_S = greedy_construct(o)
        S_set, dist_to_S = local_search(S_set, dist_to_S, max_iters=1500)
        
        if len(S_set) > len(best_S_set):
            best_S_set = S_set.copy()
            best_dist_to_S = {k: set(v) for k, v in dist_to_S.items()}
            
    # Final polish on the best set found with more aggressive search
    if time.time() < deadline - 10 and best_S_set:
        best_S_set, best_dist_to_S = local_search(best_S_set, best_dist_to_S, max_iters=5000)
        
        # Additional targeted polish: try multiple kicks and keep the best
        if time.time() < deadline - 5:
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                temp_set = set(best_S_set)
                temp_dist = {k: set(v) for k, v in best_dist_to_S.items()}
                temp_set, temp_dist = local_search(temp_set, temp_dist, max_iters=1000)
                if len(temp_set) > len(best_S_set):
                    best_S_set = temp_set
                    best_dist_to_S = temp_dist

    return [points[i] for i in best_S_set]

# For testing
if __name__ == "__main__":
    res = construct()
    print(len(res))
