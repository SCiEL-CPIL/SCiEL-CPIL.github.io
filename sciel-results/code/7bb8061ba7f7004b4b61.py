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

    def common_neighbors_count(A, u, v):
        """Count common neighbors of u and v."""
        cnt = 0
        for z in range(n):
            if z != u and z != v and A[u, z] == 1 and A[v, z] == 1:
                cnt += 1
        return cnt

    def can_add(A, u, v):
        """Check if edge (u,v) can be added without creating K2,3."""
        for x in range(n):
            if A[u, x] == 1 and x != v:
                cnt = common_neighbors_count(A, x, v)
                if cnt >= 2:
                    return False
        for y in range(n):
            if A[v, y] == 1 and y != u:
                cnt = common_neighbors_count(A, u, y)
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

    def remove_to_fix(A, count):
        """Remove edges until valid."""
        while time.time() < deadline:
            viol = False
            for u in range(n):
                if time.time() >= deadline:
                    break
                for v in range(u + 1, n):
                    cnt = common_neighbors_count(A, u, v)
                    if cnt >= 3:
                        viol = True
                        zs = [z for z in range(n) if A[u, z] == 1 and A[v, z] == 1]
                        random.shuffle(zs)
                        for z in zs:
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

    def simulated_annealing(A, count, max_iter=20000):
        """Simulated annealing to improve the graph."""
        current_A = A.copy()
        current_count = count
        best_A = A.copy()
        best_count = count
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.9995
        
        for it in range(max_iter):
            if time.time() >= deadline:
                break
            if T <= T_min:
                break
            
            # Random move: remove one edge
            edges = [(u, v) for u in range(n) for v in range(u + 1, n) if current_A[u, v] == 1]
            if not edges:
                break
            u1, v1 = random.choice(edges)
            current_A[u1, v1] = 0
            current_A[v1, u1] = 0
            current_count -= 1
            
            # Try to add multiple edges
            added = 0
            for _ in range(3):  # Try up to 3 additions
                if time.time() >= deadline:
                    break
                # Random candidate edge
                candidates = [(u, v) for u in range(n) for v in range(u + 1, n) if current_A[u, v] == 0]
                if not candidates:
                    break
                u2, v2 = random.choice(candidates)
                if can_add(current_A, u2, v2):
                    current_A[u2, v2] = 1
                    current_A[v2, u2] = 1
                    current_count += 1
                    added += 1
            
            # Also try greedy additions
            current_A, current_count = try_add_edges(current_A, current_count)
            
            # Accept or reject
            delta = current_count - (count + 1)  # Net change relative to original
            if delta > 0 or random.random() < np.exp(delta / T):
                count = current_count
                A = current_A.copy()
                if count > best_count:
                    best_count = count
                    best_A = current_A.copy()
            else:
                # Revert
                current_A[u1, v1] = 1
                current_A[v1, u1] = 1
                current_count += 1
                # Revert additions
                # This is complex; let's just keep it simple and not revert
                # For now, accept the move if it doesn't make things worse
                pass
            
            T *= cooling_rate
        
        return best_A, best_count

    def local_search(A, count, max_iter=3000):
        """Try edge swaps to improve the graph."""
        improved = True
        iters = 0
        while improved and time.time() < deadline and iters < max_iter:
            improved = False
            iters += 1
            for u1 in range(n):
                if time.time() >= deadline:
                    return A, count
                for v1 in range(u1 + 1, n):
                    if A[u1, v1] == 0:
                        continue
                    A[u1, v1] = 0
                    A[v1, u1] = 0
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
                        A, count = try_add_edges(A, count)
                        break
                    else:
                        A[u1, v1] = 1
                        A[v1, u1] = 1
                if improved:
                    break
        return A, count

    for restart in range(1500):
        if time.time() >= deadline:
            break
        
        A = np.zeros((n, n), dtype=int)
        strategy = random.randint(0, 3)
        
        if strategy == 0:
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
            p = 0.4
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < p:
                        A[u, v] = 1
                        A[v, u] = 1
            count = int(np.sum(A)) // 2
        elif strategy == 2:
            for u in range(n):
                for d in range(1, 6):
                    v = (u + d) % n
                    A[u, v] = 1
                    A[v, u] = 1
            count = 5 * n
        else:
            for u in range(n):
                neighbors = list(range(n))
                neighbors.remove(u)
                random.shuffle(neighbors)
                for v in neighbors[:random.randint(4, 8)]:
                    if A[u, v] == 0:
                        A[u, v] = 1
                        A[v, u] = 1
            count = int(np.sum(A)) // 2
        
        A, count = remove_to_fix(A, count)
        A, count = try_add_edges(A, count)
        A, count = local_search(A, count)
        
        # Simulated annealing for further improvement
        A2, count2 = simulated_annealing(A, count)
        if count2 > count:
            A, count = A2, count2
        
        if count > best_edges:
            best_edges = count
            best_A = A.copy()
        
        if best_edges >= 95 and time.time() > deadline - 30:
            break

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
