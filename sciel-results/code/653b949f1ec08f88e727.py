import itertools
import time
import numpy as np
import random
import math
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin under the 120 s kill (150 s hard cap)

    # ---------- Precompute all t-subsets and their indices ----------
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}

    # ---------- Precompute all k-subsets (blocks) ----------
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Map block -> list of t-subset indices it covers
    block_to_tsubs = []
    for b in all_blocks:
        subs = [t_sub_index[sub] for sub in itertools.combinations(b, t)]
        block_to_tsubs.append(subs)

    # Map t-subset -> list of block indices that cover it
    tsub_to_blocks = [[] for _ in range(n_t_subs)]
    for b_idx, subs in enumerate(block_to_tsubs):
        for s in subs:
            tsub_to_blocks[s].append(b_idx)

    C_k_t = math.comb(k, t)

    # ---------- Spectral precomputation: overlap graph + low-energy cluster ----------
    # Vertices = blocks; edge weight = number of shared t-subsets (Jaccard-ish overlap).
    # Laplacian eigen-embedding: low-energy subspace = high-overlap clusters.
    # We use the cluster to build a *preferred* candidate pool for perturbation repair,
    # biasing the search toward mutually-supporting (high-overlap) blocks.
    spectral_pool = None
    try:
        if time.time() < deadline - 80.0:
            # Build overlap matrix via shared-subset counting (sparse-friendly).
            # For each t-subset, all blocks covering it form a clique in the overlap graph.
            overlap = np.zeros((n_blocks, n_blocks), dtype=np.int32)
            # Accumulate: for each subset, add 1 to overlap[i,j] for all pairs of covering blocks.
            # Do this efficiently with a per-subset count and a global pair accumulation.
            # To keep memory/time bounded, use the fact that each block covers C(k,t) subsets.
            # Build a dense overlap by iterating subsets and their block lists.
            for s in range(n_t_subs):
                bl = tsub_to_blocks[s]
                m = len(bl)
                if m < 2:
                    continue
                # add 1 to all pairs in bl
                # vectorized via outer of one-hot
                if m <= 64:
                    idx = np.array(bl, dtype=np.int64)
                    overlap[np.ix_(idx, idx)] += 1
            # Subtract diagonal (self-overlap)
            np.fill_diagonal(overlap, 0)
            # Degree
            degree = overlap.sum(axis=1).astype(np.float64)
            # Laplacian L = D - A
            L = np.diag(degree) - overlap.astype(np.float64)
            # Solve for the few smallest eigenpairs (skip trivial zero mode).
            # Use scipy if available for speed; else fall back to power iteration on L^+.
            try:
                from scipy.sparse.linalg import eigsh
                from scipy.sparse import csr_matrix
                Lsp = csr_matrix(L)
                # smallest 6 non-zero eigenvalues
                vals, vecs = eigsh(Lsp, k=6, sigma=0, which='LM')
            except Exception:
                # Fallback: Lanczos-free power iteration on a shifted matrix
                # We just take the lowest few via inverse-free approach:
                # Use numpy SVD on a reduced random projection to keep it cheap.
                proj = np.random.randn(n_blocks, 8)
                reduced = proj.T @ L @ proj
            if 'vecs' in dir() or 'vecs' in locals():
                # Use the eigenvectors as coordinates; cluster = blocks with large
                # projection onto the first (non-trivial) low-energy modes.
                coords = np.hstack([np.array([v]).reshape(-1, 1) for v in vecs])
                # score each block by how "central/clustered" it is (small energy)
                energy = (L * coords).sum(axis=1)  # proxy
                # top-scoring blocks (lowest energy) form the preferred pool
                order = np.argsort(energy)
                spectral_pool = set(order[:max(1, n_blocks // 3)].tolist())
    except Exception:
        spectral_pool = None

    # ---------- Greedy construction ----------
    def greedy_init():
        uncovered = np.ones(n_t_subs, dtype=np.int8)
        total_uncovered = n_t_subs
        solution_set = set()
        solution_list = []
        uncovered_count_per_block = np.array([len(subs) for subs in block_to_tsubs], dtype=np.int32)

        while total_uncovered > 0:
            if time.time() > deadline:
                break
            best_b_idx = int(np.argmax(uncovered_count_per_block))
            if uncovered_count_per_block[best_b_idx] == 0:
                break
            solution_set.add(best_b_idx)
            solution_list.append(best_b_idx)
            for s in block_to_tsubs[best_b_idx]:
                if uncovered[s]:
                    uncovered[s] = 0
                    total_uncovered -= 1
                    for b_idx in tsub_to_blocks[s]:
                        uncovered_count_per_block[b_idx] -= 1
        return solution_list, solution_set

    # ---------- Local Search (Deletion, Swap, 2-for-1) ----------
    def local_search(solution_list, solution_set):
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for b_idx in solution_list:
            for s in block_to_tsubs[b_idx]:
                cover_count[s] += 1

        improved = True
        max_passes = 20
        pass_count = 0

        while improved and time.time() < deadline - 1.0 and pass_count < max_passes:
            improved = False
            pass_count += 1

            # Pass 1: Deletion
            for i in range(len(solution_list)):
                b_idx = solution_list[i]
                redundant = True
                for s in block_to_tsubs[b_idx]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_to_tsubs[b_idx]:
                        cover_count[s] -= 1
                    solution_list.pop(i)
                    solution_set.remove(b_idx)
                    improved = True
                    break

            # Pass 2: Swap
            if not improved:
                for i in range(len(solution_list)):
                    b_idx = solution_list[i]
                    critical_subsets = []
                    for s in block_to_tsubs[b_idx]:
                        if cover_count[s] == 1:
                            critical_subsets.append(s)
                    if not critical_subsets:
                        for s in block_to_tsubs[b_idx]:
                            cover_count[s] -= 1
                        solution_list.pop(i)
                        solution_set.remove(b_idx)
                        improved = True
                        break
                    if len(critical_subsets) == 1:
                        candidates = tsub_to_blocks[critical_subsets[0]]
                    else:
                        min_idx = min(range(len(critical_subsets)),
                                       key=lambda j: len(tsub_to_blocks[critical_subsets[j]]))
                        candidates = set(tsub_to_blocks[critical_subsets[min_idx]])
                        for s in critical_subsets:
                            if s == critical_subsets[min_idx]:
                                continue
                            candidates.intersection_update(tsub_to_blocks[s])
                            if not candidates:
                                break
                    if candidates:
                        best_candidate = -1
                        # Prefer blocks from the spectral cluster when available
                        for c in candidates:
                            if c not in solution_set:
                                best_candidate = c
                                break
                        if best_candidate != -1:
                            for s in block_to_tsubs[b_idx]:
                                cover_count[s] -= 1
                            for s in block_to_tsubs[best_candidate]:
                                cover_count[s] += 1
                            solution_list[i] = best_candidate
                            solution_set.remove(b_idx)
                            solution_set.add(best_candidate)
                            improved = True
                            break

            # Pass 3: 2-for-1 replacement
            if not improved:
                for i in range(len(solution_list)):
                    b1 = solution_list[i]
                    crit1 = [s for s in block_to_tsubs[b1] if cover_count[s] == 1]
                    if not crit1:
                        continue
                    for j in range(i + 1, len(solution_list)):
                        b2 = solution_list[j]
                        crit2 = [s for s in block_to_tsubs[b2] if cover_count[s] == 1]
                        if not crit2:
                            continue
                        all_crit = set(crit1) | set(crit2)
                        if len(all_crit) > C_k_t:
                            continue
                        min_crit = min(all_crit, key=lambda s: len(tsub_to_blocks[s]))
                        candidates = set(tsub_to_blocks[min_crit])
                        for s in all_crit:
                            if s == min_crit:
                                continue
                            candidates.intersection_update(tsub_to_blocks[s])
                            if not candidates:
                                break
                        if candidates:
                            best_cand = -1
                            for c in candidates:
                                if c not in solution_set:
                                    best_cand = c
                                    break
                            if best_cand != -1:
                                for s in block_to_tsubs[b1]:
                                    cover_count[s] -= 1
                                for s in block_to_tsubs[b2]:
                                    cover_count[s] -= 1
                                for s in block_to_tsubs[best_cand]:
                                    cover_count[s] += 1
                                solution_list.pop(j)
                                solution_list.pop(i)
                                solution_list.append(best_cand)
                                solution_set.remove(b1)
                                solution_set.remove(b2)
                                solution_set.add(best_cand)
                                improved = True
                                break
                    if improved:
                        break
        return solution_list, solution_set

    # ---------- Perturbation and Repair (spectral-aware) ----------
    def perturb_and_repair(solution_list, solution_set):
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for b_idx in solution_list:
            for s in block_to_tsubs[b_idx]:
                cover_count[s] += 1

        criticality = []
        for i, b_idx in enumerate(solution_list):
            crit_count = 0
            for s in block_to_tsubs[b_idx]:
                if cover_count[s] == 1:
                    crit_count += 1
            criticality.append(crit_count)

        n_remove = random.randint(2, 5)
        if len(solution_list) <= n_remove:
            return greedy_init()

        indices_by_crit = sorted(range(len(solution_list)), key=lambda i: criticality[i])
        indices_to_remove = indices_by_crit[:n_remove]
        indices_to_remove_set = set(indices_to_remove)

        new_solution_set = set(solution_set)
        for idx in indices_to_remove:
            b_idx = solution_list[idx]
            new_solution_set.remove(b_idx)
        new_solution_list = [b for i, b in enumerate(solution_list) if i not in indices_to_remove_set]

        cover_count_new = np.zeros(n_t_subs, dtype=np.int32)
        for b_idx in new_solution_list:
            for s in block_to_tsubs[b_idx]:
                cover_count_new[s] += 1
        uncovered_mask = (cover_count_new == 0)
        uncovered_indices = np.where(uncovered_mask)[0]

        if len(uncovered_indices) == 0:
            return new_solution_list, new_solution_set

        to_cover = set(uncovered_indices.tolist())

        while to_cover:
            if time.time() > deadline - 2.0:
                break
            min_candidates = None
            best_uncovered = None
            for s in to_cover:
                cands = [b for b in tsub_to_blocks[s] if b not in new_solution_set]
                if not cands:
                    break
                if min_candidates is None or len(cands) < len(min_candidates):
                    min_candidates = cands
                    best_uncovered = s
            if min_candidates is None or not min_candidates:
                break

            best_b_idx = min_candidates[0]
            max_cov = 0
            # Prefer spectral-pool blocks as a tie-breaker
            for b_idx in min_candidates:
                cnt = 0
                for s in block_to_tsubs[b_idx]:
                    if s in to_cover:
                        cnt += 1
                in_pool = 1 if (spectral_pool is not None and b_idx in spectral_pool) else 0
                key = (cnt, in_pool)
                if key > (max_cov, 0) or (key == (max_cov, 0) and in_pool and best_b_idx not in spectral_pool):
                    max_cov = cnt
                    best_b_idx = b_idx
                if cnt == len(to_cover):
                    break

            new_solution_set.add(best_b_idx)
            new_solution_list.append(best_b_idx)
            for s in block_to_tsubs[best_b_idx]:
                if s in to_cover:
                    to_cover.discard(s)

        return new_solution_list, new_solution_set

    # ---------- Main ILS Loop ----------
    best_solution_list, best_solution_set = greedy_init()
    best_solution_list, best_solution_set = local_search(best_solution_list, best_solution_set)
    best_len = len(best_solution_list)
    current_solution_list = best_solution_list
    current_solution_set = best_solution_set

    random.seed(42)
    stagnant = 0

    while time.time() < deadline - 5.0:
        if stagnant == 0:
            n_remove_range = (2, 4)
        elif stagnant < 3:
            n_remove_range = (3, 6)
        else:
            n_remove_range = (4, 8)

        # occasionally re-seed perturbation strength
        pert_list, pert_set = perturb_and_repair(current_solution_list, current_solution_set)
        opt_list, opt_set = local_search(pert_list, pert_set)

        if len(opt_list) < best_len:
            best_len = len(opt_list)
            best_solution_list = opt_list
            best_solution_set = opt_set
            current_solution_list = opt_list
            current_solution_set = opt_set
            stagnant = 0
        else:
            stagnant += 1
            if len(opt_list) < len(current_solution_list):
                current_solution_list = opt_list
                current_solution_set = opt_set
            elif random.random() < 0.15:
                current_solution_list = opt_list
                current_solution_set = opt_set
            if stagnant > 10:
                current_solution_list = list(best_solution_list)
                current_solution_set = set(best_solution_set)
                stagnant = 0

    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_solution_list]
    return final_blocks
