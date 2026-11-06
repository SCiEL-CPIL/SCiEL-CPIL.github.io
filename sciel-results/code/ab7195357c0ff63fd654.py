import numpy as np
import random
import time

def construct():
    n = 20
    best_edges = -1
    best_A = None
    deadline = time.time() + 110

    def is_valid(A):
        K = A @ A
        for u in range(n):
            for v in range(u + 1, n):
                if K[u, v] >= 3:
                    return False
        return True

    def edge_count(A):
        return int(np.sum(A)) // 2

    def can_add(A, u, v):
        """Check if edge (u,v) can be added without creating K2,3."""
        # Adding (u,v): for each x in N(u)\{v}, K[x,v] increases by 1
        # For each y in N(v)\{u}, K[u,y] increases by 1
        # Need K[x,v] <= 1 for all x in N(u)\{v}
        # Need K[u,y] <= 1 for all y in N(v)\{u}
        for x in range(n):
            if A[u, x] == 1 and x != v:
                # Count common neighbors of x and v
                cnt = 0
                for z in range(n):
                    if A[x, z] == 1 and A[v, z] == 1:
                        cnt += 1
                if cnt >= 2:
                    return False
        for y in range(n):
            if A[v, y] == 1 and y != u:
                # Count common neighbors of u and y
                cnt = 0
                for z in range(n):
                    if A[u, z] == 1 and A[y, z] == 1:
                        cnt += 1
                if cnt >= 2:
                    return False
        return True

    def try_add_edges(A, count):
        """Greedily add edges while maintaining validity."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for u in range(n):
                if time.time() >= deadline:
                    return A, count
                for v in range(u + 1, n):
                    if A[u, v] == 0 and can_add(A, u, v):
                        A[u, v] = 1
                        A[v, u] = 1
                        count += 1
                        changed = True
                        break
        return A, count

    def local_search(A, count, max_iter=5000):
        """Try edge swaps to improve the graph."""
        improved = True
        iters = 0
        while improved and time.time() < deadline and iters < max_iter:
            improved = False
            iters += 1
            # Try removing one edge and adding another
            for u1 in range(n):
                if time.time() >= deadline:
                    return A, count
                for v1 in range(u1 + 1, n):
                    if A[u1, v1] == 0:
                        continue
                    # Try removing (u1, v1)
                    A[u1, v1] = 0
                    A[v1, u1] = 0
                    # Now try to add multiple edges
                    added_any = False
                    for u2 in range(n):
                        if time.time() >= deadline:
                            break
                        for v2 in range(u2 + 1, n):
                            if A[u2, v2] == 0 and can_add(A, u2, v2):
                                A[u2, v2] = 1
                                A[v2, u2] = 1
                                count += 1
                                added_any = True
                                break
                        if added_any:
                            break
                    if added_any:
                        improved = True
                        # Try to add more
                        A, count = try_add_edges(A, count)
                        break
                    else:
                        # Restore
                        A[u1, v1] = 1
                        A[v1, u1] = 1
                if improved:
                    break
        return A, count

    def remove_to_fix(A, count):
        """Remove edges until valid."""
        while time.time() < deadline:
            # Find a violation
            viol = False
            for u in range(n):
                if time.time() >= deadline:
                    break
                for v in range(u + 1, n):
                    # Count common neighbors
                    cnt = 0
                    for z in range(n):
                        if A[u, z] == 1 and A[v, z] == 1:
                            cnt += 1
                    if cnt >= 3:
                        viol = True
                        # Pick a common neighbor z to remove edge (u, z) or (v, z)
                        zs = [z for z in range(n) if A[u, z] == 1 and A[v, z] == 1]
                        random.shuffle(zs)
                        for z in zs:
                            # Remove edge (u, z)
                            A[u, z] = 0
                            A[z, u] = 0
                            count -= 1
                            break
                        break
                if viol:
                    break
            if not viol:
                break
        return A, count

    # Strategy: Start from various initializations, repair, add, local search
    for restart in range(3000):
        if time.time() >= deadline:
            break
        
        # Random initialization with varying density
        A = np.zeros((n, n), dtype=int)
        
        # Choose a random initialization strategy
        strategy = random.randint(0, 3)
        
        if strategy == 0:
            # Two K10s with random permutation
            perm = list(range(n))
            random.shuffle(perm)
            for a in range(10):
                for b in range(a + 1, 10):
                    A[perm[a], perm[b]] = 1
                    A[perm[b], perm[a]] = 1
            for a in range(10, 20):
                for b in range(a + 1, 20):
                    A[perm[a], perm[b]] = 1
                    A[perm[b], perm[a]] = 1
            count = 90
        elif strategy == 1:
            # Random graph with p=0.4
            p = 0.4
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < p:
                        A[u, v] = 1
                        A[v, u] = 1
            count = int(np.sum(A)) // 2
        elif strategy == 2:
            # Cycle-based: connect each vertex to next 5 and previous 5
            for u in range(n):
                for d in range(1, 6):
                    v = (u + d) % n
                    A[u, v] = 1
                    A[v, u] = 1
            count = 5 * n
        else:
            # Random regular-ish graph
            for u in range(n):
                neighbors = list(range(n))
                neighbors.remove(u)
                random.shuffle(neighbors)
                for v in neighbors[:random.randint(4, 8)]:
                    if A[u, v] == 0:
                        A[u, v] = 1
                        A[v, u] = 1
            count = int(np.sum(A)) // 2
        
        # Repair
        A, count = remove_to_fix(A, count)
        
        # Add edges
        A, count = try_add_edges(A, count)
        
        # Local search
        A, count = local_search(A, count)
        
        if count > best_edges:
            best_edges = count
            best_A = A.copy()
        
        # Early termination if we've found a good solution
        if best_edges >= 95 and time.time() > deadline - 30:
            break

    # If best_A is None, fallback to simple construction
    if best_A is None:
        best_A = np.zeros((n, n), dtype=int)
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = 4 * k + a
                    v = 4 * k + b
                    best_A[u, v] = 1
                    best_A[v, u] = 1
        best_edges = 30

    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_A[u, v] == 1:
                edges.append([u, v])
    return edges
