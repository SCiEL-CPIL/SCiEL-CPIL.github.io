"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin before 120s kill


def _greedy_decode(adjacency, scores):
    """Fixed greedy decoder: pick vertices in descending score order, skipping neighbours."""
    n = adjacency.shape[0]
    order = np.argsort(-scores, kind='stable')
    selected = np.zeros(n, dtype=bool)
    chosen = []
    for v in order:
        if selected[v]:
            continue
        chosen.append(v)
        # Mark v and its neighbours as selected
        selected[v] = True
        nbrs = np.where(adjacency[v])[0]
        if len(nbrs) > 0:
            selected[nbrs] = True
    return chosen


def _local_search(adjacency, chosen_set, max_iter=200):
    """
    Improve the independent set via local search:
    - Try to add vertices not in the set that are not adjacent to any chosen vertex.
    - Try 1-for-1 swaps: remove a chosen vertex and add a better one if possible.
    - Try 2-for-1: remove two chosen, add up to three non-chosen.
    """
    n = adjacency.shape[0]
    chosen = list(chosen_set)
    chosen_set = set(chosen)
    chosen_arr = np.zeros(n, dtype=bool)
    for v in chosen:
        chosen_arr[v] = True

    # Precompute: for each vertex, whether it's adjacent to any chosen vertex
    # A vertex can be added if it's not chosen and not adjacent to any chosen vertex
    def can_add(v):
        if chosen_arr[v]:
            return False
        # Check if v is adjacent to any chosen vertex
        nbrs = np.where(adjacency[v])[0]
        if len(nbrs) == 0:
            return True
        return not np.any(chosen_arr[nbrs])

    # Compute initial set of addable vertices
    addable = np.zeros(n, dtype=bool)
    for v in range(n):
        if not chosen_arr[v]:
            nbrs = np.where(adjacency[v])[0]
            if len(nbrs) == 0 or not np.any(chosen_arr[nbrs]):
                addable[v] = True

    improved = True
    it = 0
    while improved and it < max_iter:
        it += 1
        if time.time() > _DEADLINE:
            break
        improved = False

        # Phase 1: Add any addable vertex (greedy: add all possible)
        added_any = False
        addable_list = np.where(addable)[0]
        for v in addable_list:
            if chosen_arr[v] or not addable[v]:
                continue
            # Double check
            nbrs = np.where(adjacency[v])[0]
            if len(nbrs) > 0 and np.any(chosen_arr[nbrs]):
                addable[v] = False
                continue
            chosen_arr[v] = True
            chosen.append(v)
            added_any = True
            # Update addable: neighbours of v can no longer be added
            if len(nbrs) > 0:
                addable[nbrs] = False
            addable[v] = False
        if added_any:
            improved = True

        # Phase 2: Try swaps - for each chosen vertex, see if removing it allows adding more
        # This is expensive; do a limited version
        if len(chosen) > 0:
            # Try removing one vertex and see if we can add 2+
            # Sample a few candidates to keep it fast
            import random
            candidates = list(range(len(chosen)))
            random.shuffle(candidates)
            num_tries = min(20, len(candidates))
            for idx in candidates[:num_tries]:
                if time.time() > _DEADLINE:
                    break
                v = chosen[idx]
                # Temporarily remove v
                chosen_arr[v] = False
                chosen.remove(v)

                # Recompute addable quickly
                new_addable_count = 0
                new_addable_list = []
                for u in range(n):
                    if chosen_arr[u]:
                        continue
                    nbrs_u = np.where(adjacency[u])[0]
                    if len(nbrs_u) == 0 or not np.any(chosen_arr[nbrs_u]):
                        new_addable_count += 1
                        new_addable_list.append(u)

                if new_addable_count >= 2:
                    # Add all addable
                    for u in new_addable_list:
                        chosen_arr[u] = True
                        chosen.append(u)
                        nbrs_u = np.where(adjacency[u])[0]
                        if len(nbrs_u) > 0:
                            pass  # We'll recompute later
                    improved = True
                else:
                    # Put v back
                    chosen_arr[v] = True
                    chosen.append(v)

    return chosen


def priority(adjacency):
    """
    Assign scores to vertices to maximize the size of the greedy independent set.
    
    Strategy:
    1. Use a combination of degree and local structure to create initial scores.
    2. Run local search on the resulting independent set to improve it.
    3. The scores are designed so that the greedy decoder produces a good starting set,
       which is then improved by local search.
    
    However, the interface only returns scores - the greedy decoder is fixed.
    So we need to design scores that directly lead to a large independent set.
    
    Key insight: The greedy decoder picks highest-score first, skipping neighbors.
    To get a large independent set, we want to pick "spread out" vertices.
    
    Approach: Use a multi-round scoring:
    - Round 1: Score by -degree (low degree first) - baseline
    - Round 2: After simulating the greedy pick, identify which vertices were "wasted"
      (high-degree vertices that blocked many others). Adjust scores.
    
    Actually, let's try a different structural approach:
    - Use the eigenvalue/eigenvector of the adjacency matrix for a spectral scoring.
    - Or use a centrality measure that captures "independence" potential.
    
    Let's try: score = -degree + alpha * (something that rewards being far from other high-degree nodes)
    
    A better idea: Simulate the greedy process and use the result to inform better scores.
    But we only get one shot at returning scores.
    
    Let's use a two-phase approach within the time budget:
    1. Compute initial scores using a refined heuristic.
    2. Run the greedy decode.
    3. Run local search to improve.
    4. But wait - we can't change the decoder. We only return scores.
    
    So the scores must be such that the FIXED greedy decoder produces a good set.
    
    New approach: Use iterative refinement of scores.
    - Start with -degree scores.
    - Simulate greedy decode.
    - For vertices that were skipped due to being neighbors of picked vertices,
      check if they could have been picked instead with better global effect.
    - Adjust scores accordingly.
    - Repeat a few times.
    
    This is essentially trying to find scores that make the greedy algorithm behave
    like a better algorithm.
    """
    n = adjacency.shape[0]
    
    # Phase 1: Initial scoring using degree + second-order structure
    deg = adjacency.sum(axis=1).astype(float)
    
    # Second-order: for each vertex, sum of degrees of its neighbors
    # Vertices with low-degree neighbors are "better" to pick
    if n > 0:
        nbr_deg_sum = adjacency @ deg
        # Normalize to avoid issues
        nbr_deg_sum = nbr_deg_sum / (deg + 1e-10)
    else:
        nbr_deg_sum = np.zeros(n)
    
    # Initial score: prefer low degree and low neighbor-degree-sum
    scores = -deg - 0.5 * nbr_deg_sum
    
    # Phase 2: Iterative refinement
    # Simulate greedy, identify conflicts, adjust scores
    for iteration in range(10):
        if time.time() > _DEADLINE - 5.0:
            break
        
        # Simulate greedy decode
        order = np.argsort(-scores, kind='stable')
        selected = np.zeros(n, dtype=bool)
        picked = []
        for v in order:
            if selected[v]:
                continue
            picked.append(v)
            selected[v] = True
            nbrs = np.where(adjacency[v])[0]
            if len(nbrs) > 0:
                selected[nbrs] = True
        
        # For each picked vertex, compute how many vertices it "blocked"
        # If a picked vertex blocked many high-potential vertices, maybe we should have picked differently
        # Heuristic: boost scores of vertices that are far from picked vertices
        
        # Compute distance-based boost
        # Vertices not picked and not blocked get a small boost
        # Actually, let's try: for each vertex not picked, if it's not adjacent to any picked,
        # it means it could have been added. Boost such vertices.
        
        not_picked_not_blocked = np.zeros(n, dtype=bool)
        for v in range(n):
            if selected[v]:
                continue
            # v was not selected, meaning it's a neighbor of some picked vertex
            # Check if v is actually adjacent to a picked vertex
            nbrs_v = np.where(adjacency[v])[0]
            if len(nbrs_v) == 0:
                not_picked_not_blocked[v] = True
            else:
                # Check if any neighbor is picked
                is_blocked = False
                for u in nbrs_v:
                    if u in picked:  # This is O(n^2) worst case, let's use array
                        is_blocked = True
                        break
                if not is_blocked:
                    not_picked_not_blocked[v] = True
        
        # Use array-based check for speed
        picked_arr = np.zeros(n, dtype=bool)
        for v in picked:
            picked_arr[v] = True
        
        not_picked_not_blocked = np.zeros(n, dtype=bool)
        for v in range(n):
            if selected[v]:
                continue
            nbrs_v = np.where(adjacency[v])[0]
            if len(nbrs_v) == 0:
                not_picked_not_blocked[v] = True
            else:
                if not np.any(picked_arr[nbrs_v]):
                    not_picked_not_blocked[v] = True
        
        # Boost vertices that were not blocked (could have been added)
        # This encourages the greedy to pick more "independent" vertices
        scores[not_picked_not_blocked] += 0.1 * (n - len(picked)) / max(n, 1)
        
        # Also, penalize vertices that blocked many others
        for v in picked:
            nbrs = np.where(adjacency[v])[0]
            blocked_count = len(nbrs)
            scores[v] -= 0.05 * blocked_count
    
    # Final cleanup: ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
