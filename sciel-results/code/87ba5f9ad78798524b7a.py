import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0

    # Generate all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    total_words = len(all_words)
    candidates = all_words
    num_candidates = len(candidates)

    # Precompute coverage matrix
    coverage = np.zeros((num_candidates, total_words), dtype=bool)
    chunk_size = 256
    for i_start in range(0, num_candidates, chunk_size):
        i_end = min(i_start + chunk_size, num_candidates)
        diff = candidates[i_start:i_end, None, :] != all_words[None, :, :]
        dists = np.sum(diff, axis=2, dtype=np.int16)
        coverage[i_start:i_end, :] = dists <= radius

    # Geometric projection: create 2D tile structure
    # Use two linear projections to map 7D ternary space to 2D
    # Projection 1: sum of positions 0,2,4,6 mod 3
    # Projection 2: sum of positions 1,3,5 mod 3
    proj1 = (all_words[:, 0] + all_words[:, 2] + all_words[:, 4] + all_words[:, 6]) % 3
    proj2 = (all_words[:, 1] + all_words[:, 3] + all_words[:, 5]) % 3
    tile_id = proj1 * 3 + proj2  # 0..8, 9 tiles

    # Count words per tile
    tile_counts = np.zeros(9, dtype=np.int32)
    for t in range(9):
        tile_counts[t] = np.sum(tile_id == t)

    # Inverse tile density weight: words in sparse tiles get higher weight
    tile_weight = 1.0 / (tile_counts + 1)
    word_weight = tile_weight[tile_id]  # shape (total_words,)

    # Weighted greedy set cover
    remaining = np.ones(total_words, dtype=np.int8)
    remaining_count = total_words
    selected_indices = []

    # Precompute weighted coverage counts for greedy
    # weighted_counts[i] = sum of word_weight[j] for j covered by i and remaining
    # We'll compute this incrementally

    while remaining_count > 0:
        if time.time() > deadline - 30.0:
            break

        # Compute weighted coverage for each candidate
        # For each candidate i: sum of word_weight[j] where coverage[i,j] and remaining[j]
        # Vectorized: coverage @ (word_weight * remaining)
        effective_weight = word_weight * remaining
        weighted_counts = coverage.dot(effective_weight)

        best_idx = np.argmax(weighted_counts)
        best_count = weighted_counts[best_idx]

        if best_count <= 0:
            break

        selected_indices.append(best_idx)
        remaining[coverage[best_idx]] = 0
        remaining_count = np.count_nonzero(remaining)

    # Check if fully covered; if not, add remaining uncovered words directly
    if remaining_count > 0:
        uncovered = np.where(remaining > 0)[0]
        # Add codewords that cover uncovered words
        for w in uncovered:
            # Find a candidate that covers w
            coverers = np.where(coverage[:, w])[0]
            if len(coverers) > 0:
                # Pick the one that covers most remaining
                best_c = coverers[np.argmax(coverage[coverers].dot(remaining))]
                if best_c not in selected_indices:
                    selected_indices.append(best_c)
                remaining[coverage[best_c]] = 0
                remaining_count = np.count_nonzero(remaining)
                if remaining_count == 0:
                    break

    current_selected = np.array(selected_indices)
    current_size = len(current_selected)

    # Local search with multiple rounds
    in_set = np.zeros(num_candidates, dtype=bool)
    in_set[current_selected] = True

    cover_counts = np.zeros(total_words, dtype=np.int16)
    for idx in current_selected:
        cover_counts[coverage[idx]] += 1

    best_size = current_size
    best_in_set = in_set.copy()

    # Multi-round local search
    for round_num in range(20):
        if time.time() > deadline - 5.0:
            break

        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False

            # Phase 1: Remove redundant codewords
            to_remove = []
            for i in np.where(in_set)[0]:
                if not np.any(coverage[i] & (cover_counts == 1)):
                    to_remove.append(i)

            if to_remove:
                for i in to_remove:
                    in_set[i] = False
                    cover_counts[coverage[i]] -= 1
                best_size = np.sum(in_set)
                improved = True
                continue

            # Phase 2: Try swaps guided by tile structure
            # Find critical codewords (those covering words with count 1)
            critical_indices = []
            for i in np.where(in_set)[0]:
                if np.any(coverage[i] & (cover_counts == 1)):
                    critical_indices.append(i)

            if not critical_indices:
                break

            # For each critical codeword, try to find a better replacement
            # Prioritize codewords in sparse tiles first
            critical_with_tile = []
            for i in critical_indices:
                # Get the tile of the essential words
                essential = np.where(coverage[i] & (cover_counts == 1))[0]
                if len(essential) > 0:
                    # Weight by inverse tile density of essential words
                    tile_w = np.mean(tile_weight[tile_id[essential]])
                    critical_with_tile.append((i, tile_w))

            # Sort by tile weight (sparse tiles first)
            critical_with_tile.sort(key=lambda x: -x[1])

            swap_found = False
            for i, _ in critical_with_tile[:30]:  # Limit to top 30
                if time.time() > deadline - 5.0:
                    break

                essential_words = np.where(coverage[i] & (cover_counts == 1))[0]
                if len(essential_words) == 0:
                    continue

                # Find candidates not in set that cover all essential words
                potential_j = np.where(coverage[:, essential_words].all(axis=1) & ~in_set)[0]
                if len(potential_j) == 0:
                    continue

                # Score potential swaps
                best_j = -1
                best_score = -1e10

                # Limit to top candidates by coverage
                total_cov = np.sum(coverage[potential_j], axis=1)
                sorted_idx = np.argsort(-total_cov)

                for idx in sorted_idx[:40]:
                    j = potential_j[idx]

                    # Simulate swap
                    new_cover_counts = cover_counts.copy()
                    new_cover_counts[coverage[i]] -= 1
                    new_cover_counts[coverage[j]] += 1

                    # Check validity
                    if np.any(new_cover_counts < 0):
                        continue

                    # Score: minimize number of essential words, maximize coverage
                    num_essentials = np.sum(new_cover_counts == 1)
                    score = -num_essentials * 10.0 + total_cov[idx]

                    if score > best_score:
                        best_score = score
                        best_j = j

                if best_j != -1 and best_score > -50:
                    # Perform swap
                    in_set[i] = False
                    in_set[best_j] = True
                    cover_counts[coverage[i]] -= 1
                    cover_counts[coverage[j]] += 1
                    improved = True
                    swap_found = True
                    break

            if not swap_found:
                break

        # Update best
        current_size = np.sum(in_set)
        if current_size < best_size:
            best_size = current_size
            best_in_set = in_set.copy()

    # Use best solution found
    in_set = best_in_set
    final_indices = np.where(in_set)[0]
    selected_codewords = [candidates[i].tolist() for i in final_indices]

    return selected_codewords
