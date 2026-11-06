<<<<<<< SEARCH
def _local_search_hypergraph(points, deadline):
    """
    Local search optimized for constraint satisfaction:
    - Try moving points to increase perfect triple count
    - Maintain no-four-collinear constraint
    - Use hypergraph matching perspective
    """
    best = [list(p) for p in points]
    lines = _build_lines(best)
    best_triples = _count_perfect_triples(lines)
    best_score = _compute_score(best, deadline)
    
    for iteration in range(50):
        if time.time() > deadline - 10:
            break
        improved = False
        for i in range(20):
            if time.time() > deadline - 10:
                break
            orig = best[i][:]
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = orig[0] + dx, orig[1] + dy
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                    if (nx, ny) in [tuple(p) for p in best if p is not best[i]]:
                        continue
                    candidate = [list(p) for p in best]
                    candidate[i] = [nx, ny]
                    if _is_valid(candidate, deadline):
                        cand_lines = _build_lines(candidate)
                        cand_triples = _count_perfect_triples(cand_lines)
                        # Prefer more perfect triples, then lower score
                        if cand_triples > best_triples or (cand_triples == best_triples and _compute_score(candidate, deadline) < best_score):
                            best_score = _compute_score(candidate, deadline)
                            best_triples = cand_triples
                            best = candidate
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best, best_score
=======
def _local_search_hypergraph(points, deadline):
    """
    Local search optimized for constraint satisfaction:
    - Try moving points to increase pairwise conflict count (surrogate for low independence number)
    - Maintain no-four-collinear constraint
    - Periodically verify with exact score
    """
    best = [list(p) for p in points]
    lines = _build_lines(best)
    best_triples = _count_perfect_triples(lines)
    best_conflicts = _count_pairwise_conflicts(best)
    best_score = _compute_score(best, deadline)
    
    for iteration in range(80):
        if time.time() > deadline - 15:
            break
        improved = False
        rng = random.Random(iteration)
        # Randomize point order to avoid local optima
        order = list(range(20))
        rng.shuffle(order)
        for i in order:
            if time.time() > deadline - 15:
                break
            orig = best[i][:]
            # Try larger moves occasionally
            max_dx = 12 if iteration % 3 == 0 else 8
            for dx in range(-max_dx, max_dx + 1):
                if time.time() > deadline - 15:
                    break
                for dy in range(-max_dx, max_dx + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = orig[0] + dx, orig[1] + dy
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                    if (nx, ny) in [tuple(p) for p in best if p is not best[i]]:
                        continue
                    candidate = [list(p) for p in best]
                    candidate[i] = [nx, ny]
                    if _is_valid(candidate, deadline):
                        cand_lines = _build_lines(candidate)
                        cand_triples = _count_perfect_triples(cand_lines)
                        cand_conflicts = _count_pairwise_conflicts(candidate)
                        # Prefer more conflicts, then more perfect triples, then lower score
                        if (cand_conflicts > best_conflicts or
                            (cand_conflicts == best_conflicts and cand_triples > best_triples) or
                            (cand_conflicts == best_conflicts and cand_triples == best_triples and _compute_score(candidate, deadline) < best_score)):
                            best_score = _compute_score(candidate, deadline)
                            best_triples = cand_triples
                            best_conflicts = cand_conflicts
                            best = candidate
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            # Try a random restart from current best
            if time.time() > deadline - 20:
                break
            for _ in range(5):
                if time.time() > deadline - 20:
                    break
                i = rng.randint(0, 19)
                for _ in range(30):
                    dx = rng.randint(-15, 15)
                    dy = rng.randint(-15, 15)
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = best[i][0] + dx, best[i][1] + dy
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                    if (nx, ny) in [tuple(p) for p in best if p is not best[i]]:
                        continue
                    candidate = [list(p) for p in best]
                    candidate[i] = [nx, ny]
                    if _is_valid(candidate, deadline):
                        cand_lines = _build_lines(candidate)
                        cand_triples = _count_perfect_triples(cand_lines)
                        cand_conflicts = _count_pairwise_conflicts(candidate)
                        if (cand_conflicts > best_conflicts or
                            (cand_conflicts == best_conflicts and cand_triples > best_triples)):
                            best_score = _compute_score(candidate, deadline)
                            best_triples = cand_triples
                            best_conflicts = cand_conflicts
                            best = candidate
                            improved = True
                            break
                if improved:
                    break
    
    return best, best_score
>>>>>>> REPLACE

