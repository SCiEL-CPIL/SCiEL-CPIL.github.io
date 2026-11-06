"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    n = 40
    target_edges = 401

    # Representation: numpy adjacency matrix A (int32)
    # Book of edge (u,v) = |N(u) ∩ N(v)| = (A @ A)[u,v] - 1 (for an edge)

    def compute_max_book(A):
        A2 = A @ A
        mx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] == 1:
                    b = int(A2[i, j]) - 1
                    if b > mx:
                        mx = b
        return mx

    def get_worst_edges(A, A2, k=10):
        """Return up to k edges with the largest book, sorted descending by book."""
        edge_books = []
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] == 1:
                    b = int(A2[i, j]) - 1
                    edge_books.append((b, i, j))
        edge_books.sort(reverse=True)
        return edge_books[:k]

    def check_affected_max(A, A2, vertices):
        """Compute max book among edges incident to any vertex in 'vertices'."""
        mx = 0
        for u in vertices:
            for v in range(n):
                if u != v and A[u, v] == 1:
                    b = int(A2[u, v]) - 1
                    if b > mx:
                        mx = b
        return mx

    # ---- Initial construction: balanced near-bipartite with intra edges spread out ----
    # Start with K20,20 (400 edges) + 1 intra edge.
    # But that gives book 20 for the intra edge and 0 for cross edges.
    # We want to break large books. Let's start with a random dense graph and anneal.
    random.seed(42)

    # Build a random graph with 401 edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_edges)
    A = np.zeros((n, n), dtype=np.int32)
    for u, v in all_edges[:target_edges]:
        A[u, v] = 1
        A[v, u] = 1

    current_max = compute_max_book(A)
    best_A = A.copy()
    best_max = current_max

    # ---- Phase 1: Simulated annealing with book-weighted edge removal ----
    T = 5.0
    cooling = 0.9997
    no_improve = 0
    max_no_improve = 8000

    A2 = A @ A

    while time.time() - start_time < DEADLINE - 20:
        if no_improve > max_no_improve and T < 1.0:
            break

        # Update A2 periodically or after each move
        A2 = A @ A

        # Get worst edges
        worst = get_worst_edges(A, A2, k=15)
        if not worst:
            break

        # Pick edge to remove: biased toward high-book edges
        if random.random() < 0.75:
            # Weighted sample from worst edges
            weights = [b + 1 for b, _, _ in worst]
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0
            remove_u = remove_v = None
            for b, i, j in worst:
                cum += weights[weights.index(b + 1)] if False else 0
            # Simpler: use numpy random choice
            idx = np.random.choice(len(worst), p=np.array(weights) / total_w)
            remove_u, remove_v = worst[idx][1], worst[idx][2]
        else:
            # Random edge
            idx = random.randint(0, target_edges - 1)
            # Find idx-th edge
            cnt = 0
            remove_u = remove_v = None
            for i in range(n):
                for j in range(i + 1, n):
                    if A[i, j] == 1:
                        if cnt == idx:
                            remove_u, remove_v = i, j
                            break
                        cnt += 1
                if remove_u is not None:
                    break

        if remove_u is None:
            continue

        # Pick a random non-edge to add
        added = False
        for _ in range(40):
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            if a == b:
                continue
            ka, kb = min(a, b), max(a, b)
            if A[ka, kb] == 1:
                continue

            # Apply move
            A[remove_u, remove_v] = 0
            A[remove_v, remove_u] = 0
            A[ka, kb] = 1
            A[kb, ka] = 1

            # Recompute A2
            A2_new = A @ A
            affected = {remove_u, remove_v, ka, kb}
            new_max = check_affected_max(A, A2_new, affected)

            # Safety: full check occasionally
            if random.random() < 0.05:
                full_max = compute_max_book(A)
                new_max = max(new_max, full_max)

            delta = new_max - current_max
            if delta <= 0:
                current_max = new_max
                A2 = A2_new
                if new_max < best_max:
                    best_max = new_max
                    best_A = A.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                added = True
                break
            else:
                if random.random() < np.exp(-delta / max(T, 1e-6)):
                    current_max = new_max
                    A2 = A2_new
                    no_improve += 1
                    added = True
                    break
                else:
                    # Revert
                    A[remove_u, remove_v] = 1
                    A[remove_v, remove_u] = 1
                    A[ka, kb] = 0
                    A[kb, ka] = 0

        if not added:
            no_improve += 1

        T *= cooling
        if T < 0.1:
            T = 0.1

    # ---- Phase 2: Targeted local search on best_A ----
    # Focus on breaking the worst books by systematically trying edge swaps
    A = best_A.copy()
    current_max = compute_max_book(A)
    best_A2 = A.copy()
    best_max2 = current_max

    A2 = A @ A

    while time.time() - start_time < DEADLINE - 5:
        A2 = A @ A
        worst = get_worst_edges(A, A2, k=5)
        if not worst:
            break
        max_book_val = worst[0][0]
        if max_book_val <= 0:
            break

        improved = False
        for b_val, u, v in worst:
            if b_val < max_book_val:
                break

            # Common neighbors of u and v
            common = np.where((A[u] == 1) & (A[v] == 1))[0]
            common = [w for w in common if w != u and w != v]

            if not common:
                continue

            # Try removing edges (u, w) or (v, w) for w in common
            for w in random.sample(common, min(len(common), 15)):
                for ru, rv in [(u, w), (v, w)]:
                    if A[ru, rv] != 1:
                        continue

                    # Find best edge to add
                    best_add = None
                    best_new_max = max_book_val

                    # Sample candidate non-edges
                    candidates = []
                    for _ in range(60):
                        a = random.randint(0, n - 1)
                        b = random.randint(0, n - 1)
                        if a == b:
                            continue
                        ka, kb = min(a, b), max(a, b)
                        if A[ka, kb] == 0:
                            candidates.append((a, b))

                    for a, b in candidates:
                        ka, kb = min(a, b), max(a, b)

                        # Apply
                        A[ru, rv] = 0
                        A[rv, ru] = 0
                        A[ka, kb] = 1
                        A[kb, ka] = 1

                        A2_t = A @ A
                        affected = {ru, rv, ka, kb}
                        new_max = check_affected_max(A, A2_t, affected)

                        # Revert
                        A[ru, rv] = 1
                        A[rv, ru] = 1
                        A[ka, kb] = 0
                        A[kb, ka] = 0

                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_add = (a, b)

                    if best_add and best_new_max < max_book_val:
                        a, b = best_add
                        ka, kb = min(a, b), max(a, b)

                        # Apply for real
                        A[ru, rv] = 0
                        A[rv, ru] = 0
                        A[ka, kb] = 1
                        A[kb, ka] = 1

                        current_max = best_new_max
                        if current_max < best_max2:
                            best_max2 = current_max
                            best_A2 = A.copy()
                        improved = True
                        break

                if improved:
                    break

            if improved:
                break

        if not improved:
            # No improvement in this pass; try a few random perturbations
            for _ in range(10):
                # Random edge swap
                edge_list = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] == 1]
                if not edge_list:
                    break
                ru, rv = random.choice(edge_list)
                for _ in range(20):
                    a = random.randint(0, n - 1)
                    b = random.randint(0, n - 1)
                    if a == b:
                        continue
                    ka, kb = min(a, b), max(a, b)
                    if A[ka, kb] == 0:
                        A[ru, rv] = 0
                        A[rv, ru] = 0
                        A[ka, kb] = 1
                        A[kb, ka] = 1
                        new_max = compute_max_book(A)
                        if new_max < current_max:
                            current_max = new_max
                            if current_max < best_max2:
                                best_max2 = current_max
                                best_A2 = A.copy()
                        else:
                            # Revert
                            A[ru, rv] = 1
                            A[rv, ru] = 1
                            A[ka, kb] = 0
                            A[kb, ka] = 0
                        break
            break  # Exit local search if stuck

    # Use the best found
    final_A = best_A2 if best_max2 < best_max else best_A
    final_max = min(best_max, best_max2)

    # Extract edges
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if final_A[i, j] == 1:
                edge_list.append([i, j])

    # Ensure exactly 401 edges
    assert len(edge_list) == target_edges, f"Edge count {len(edge_list)} != {target_edges}"

    return edge_list
# EVOLVE-BLOCK-END
