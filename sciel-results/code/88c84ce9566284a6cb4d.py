import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s kill

    N = 64
    M = N * N

    # Precompute all points as (x, y)
    points = [(x, y) for x in range(N) for y in range(N)]
    coords = np.array(points, dtype=np.int32)

    # Precompute squared distance matrix
    # diff[i,j,k] = coords[i,k] - coords[j,k]
    # dist[i,j] = sum_k diff[i,j,k]^2
    # Shape: (M, M) of int32
    diff = coords[:, None, :] - coords[None, :, :]
    dist_mat = np.sum(diff**2, axis=-1).astype(np.int32)
    del diff

    # Strategy:
    # 1. Maintain for each point in S, a set of distances to other points in S.
    # 2. To check if candidate p can be added:
    #    a) Compute distances from p to all q in S. If any repeats, fail.
    #    b) For each q in S, check if dist(q, p) is already in dists_to_others[q].
    #       If yes, then there exists r in S with dist(q,r) == dist(q,p), making (p,q,r) isosceles.
    #    This is O(|S|) per candidate!

    def try_add(S, dists_to_others, p):
        """
        Check if p can be added to S.
        dists_to_others[s] = set of squared distances from s to other points in S.
        Returns True if p can be added, False otherwise.
        """
        # Step 1: Check for duplicate distances from p to points in S
        seen_dists = set()
        for q in S:
            d = dist_mat[p, q]
            if d in seen_dists:
                return False
            seen_dists.add(d)

        # Step 2: For each q in S, check if d(p,q) is already a distance from q to some other point in S
        for q in S:
            d_pq = dist_mat[p, q]
            if d_pq in dists_to_others[q]:
                return False

        return True

    def add_point(S, dists_to_others, p):
        """Add point p to S and update dists_to_others."""
        S.add(p)
        dists_to_others[p] = set()
        for q in S:
            if q == p:
                continue
            d = dist_mat[p, q]
            dists_to_others[p].add(d)
            dists_to_others[q].add(d)

    def remove_point(S, dists_to_others, p):
        """Remove point p from S and update dists_to_others."""
        for q in S:
            if q == p:
                continue
            d = dist_mat[p, q]
            dists_to_others[q].discard(d)
        del dists_to_others[p]
        S.discard(p)

    def greedy_construct():
        S = set()
        dists_to_others = {}
        candidates = list(range(M))
        random.shuffle(candidates)

        for p in candidates:
            if time.time() > deadline:
                break
            if try_add(S, dists_to_others, p):
                add_point(S, dists_to_others, p)

        return S, dists_to_others

    def local_search(S, dists_to_others):
        S = set(S)
        dists_to_others = {k: set(v) for k, v in dists_to_others.items()}
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            S_list = list(S)
            random.shuffle(S_list)

            # Try to add points not in S
            candidates = [i for i in range(M) if i not in S]
            random.shuffle(candidates)

            for p in candidates:
                if time.time() > deadline - 5:
                    break
                if try_add(S, dists_to_others, p):
                    add_point(S, dists_to_others, p)
                    improved = True
                    break

            if improved:
                continue

            # If no addition, try a swap
            if S_list:
                q_to_remove = random.choice(S_list)
                remove_point(S, dists_to_others, q_to_remove)

                candidates = [i for i in range(M) if i not in S]
                random.shuffle(candidates)

                added = False
                for p in candidates:
                    if time.time() > deadline - 5:
                        break
                    if p == q_to_remove:
                        continue
                    if try_add(S, dists_to_others, p):
                        add_point(S, dists_to_others, p)
                        added = True
                        break

                if not added:
                    # Revert
                    add_point(S, dists_to_others, q_to_remove)
                    break
            else:
                break

        return S, dists_to_others

    best_S = set()
    best_dists = {}

    # Multiple restarts
    num_restarts = 30
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break

        S, dists = greedy_construct()
        S, dists = local_search(S, dists)

        if len(S) > len(best_S):
            best_S = S.copy()
            best_dists = {k: set(v) for k, v in dists.items()}

    # Final polish on the best set found
    if time.time() < deadline - 3 and len(best_S) > 0:
        best_S, best_dists = local_search(best_S, best_dists)

    return [points[i] for i in best_S]

# For testing
if __name__ == "__main__":
    res = construct()
    print(len(res))
