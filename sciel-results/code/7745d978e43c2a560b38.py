"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill

    d = 7
    n = 2 ** d

    # Precompute all edges as (u, v, bit) where u < v
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y, i))

    # Precompute for each edge the list of (other_edge1, other_edge2, other_edge3)
    # such that {edge, oe1, oe2, oe3} forms a C4.
    edge_cycles = {}  # (x,y) -> list of tuples of 3 other edges
    for (x, y, bit_i) in all_edges:
        cycles = []
        for j in range(d):
            if j == bit_i:
                continue
            c = x ^ (1 << j)
            e = y ^ (1 << j)
            oe1 = (min(x ^ (1 << j), y ^ (1 << j)), max(x ^ (1 << j), y ^ (1 << j)))
            oe2 = (min(x, c), max(x, c))
            oe3 = (min(y, e), max(y, e))
            cycles.append((oe1, oe2, oe3))
        edge_cycles[(x, y)] = cycles

    def is_valid_add(chosen, edge):
        """Check if adding edge to chosen creates a C4."""
        x, y, bit_i = edge
        for (oe1, oe2, oe3) in edge_cycles[(x, y)]:
            if oe1 in chosen and oe2 in chosen and oe3 in chosen:
                return False
        return True

    def greedy_randomized(seed):
        """Run greedy with randomized edge order."""
        rng = random.Random(seed)
        edges = list(all_edges)
        rng.shuffle(edges)

        chosen = set()
        out = []

        for edge in edges:
            x, y, bit_i = edge
            if is_valid_add(chosen, edge):
                chosen.add((x, y))
                out.append([x, y])

        return out, chosen

    def try_add_edges(chosen, out, rng, max_attempts=20):
        """Try to add as many edges as possible from random candidates."""
        added = 0
        for _ in range(max_attempts):
            candidate = rng.choice(all_edges)
            cx, cy, cbit = candidate
            if (cx, cy) not in chosen and is_valid_add(chosen, candidate):
                chosen.add((cx, cy))
                out.append([cx, cy])
                added += 1
        return added

    def systematic_add(chosen, out):
        """Try to add all possible edges systematically."""
        added = 0
        for edge in all_edges:
            x, y, bit_i = edge
            if (x, y) not in chosen and is_valid_add(chosen, edge):
                chosen.add((x, y))
                out.append([x, y])
                added += 1
        return added

    def local_search(out, chosen, time_remaining, rng):
        """
        Local search combining random additions, 1-2 swaps, and kicks.
        """
        no_improve_count = 0
        best_size = len(out)
        
        # Parameters
        add_attempts = 10
        kick_threshold = 30
        kick_size = 4

        while time.time() - start_time < time_limit - 3.0:
            # 1. Try to add random edges
            if try_add_edges(chosen, out, rng, max_attempts=add_attempts) > 0:
                no_improve_count = 0
                if len(out) > best_size:
                    best_size = len(out)
                continue

            # 2. Try 1-remove, 2-add swap
            idx = rng.randint(0, len(out) - 1)
            ex, ey = out[idx]
            edge_key = (ex, ey)

            # Remove the edge
            chosen.discard(edge_key)
            out.pop(idx)
            
            original_len = len(out)
            
            # Try to add up to 2 edges
            added_count = 0
            for _ in range(2):
                # Try a few random candidates to find a valid add
                found = False
                for _ in range(50):
                    candidate = rng.choice(all_edges)
                    cx, cy, cbit = candidate
                    if (cx, cy) not in chosen and is_valid_add(chosen, candidate):
                        chosen.add((cx, cy))
                        out.append([cx, cy])
                        added_count += 1
                        found = True
                        break
                if not found:
                    break

            # Evaluate the move
            if len(out) > original_len:
                # Improvement
                no_improve_count = 0
                if len(out) > best_size:
                    best_size = len(out)
            elif len(out) == original_len:
                # Neutral move, might help escape, but let's be conservative
                # Keep it if we are stuck a lot
                if no_improve_count > 10:
                    no_improve_count = 0
                else:
                    # Revert to avoid cycling
                    # Remove the added edges
                    # Note: out has 1 more edge than original_len if added_count >= 1?
                    # Wait.
                    # Start: L
                    # Remove 1: L-1
                    # Add K: L-1+K
                    # If L-1+K == L => K=1.
                    # If L-1+K > L => K >= 2.
                    
                    # If K=1, we are neutral.
                    # If K=0, we lost 1.
                    
                    # Let's just revert if not strictly better to keep it simple and fast,
                    # unless we are very stuck.
                    if added_count < 2:
                         # Revert: remove added edges
                         for _ in range(added_count):
                             last_edge = out.pop()
                             chosen.discard((last_edge[0], last_edge[1]))
                         # Re-add removed edge
                         chosen.add(edge_key)
                         out.append([ex, ey])
                         no_improve_count += 1
                    else:
                         # K>=2, so len(out) > original_len. This case is handled above.
                         pass
            else:
                # Loss
                # Revert
                for _ in range(added_count):
                    last_edge = out.pop()
                    chosen.discard((last_edge[0], last_edge[1]))
                chosen.add(edge_key)
                out.append([ex, ey])
                no_improve_count += 1

            # 3. Kick mechanism
            if no_improve_count > kick_threshold:
                # Remove random edges
                remove_count = min(kick_size, len(out))
                removed = []
                for _ in range(remove_count):
                    idx = rng.randint(0, len(out) - 1)
                    removed.append(out.pop(idx))
                    chosen.discard((removed[-1][0], removed[-1][1]))
                
                # Re-add systematically
                systematic_add(chosen, out)
                
                no_improve_count = 0
                if len(out) > best_size:
                    best_size = len(out)

        return out, chosen

    # Run multiple randomized greedy restarts
    best_result = None
    best_count = 0

    # Use more restarts
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 5.0:
            break

        rng = random.Random(restart * 12345 + 67890)
        out, chosen = greedy_randomized(restart)

        if len(out) > best_count:
            best_count = len(out)
            best_result = [list(e) for e in out]

        # Local search improvement
        if time.time() - start_time < time_limit - 5.0:
            out, chosen = local_search(out, chosen, time_limit - time.time() - start_time, rng)
            if len(out) > best_count:
                best_count = len(out)
                best_result = [list(e) for e in out]

    # Final polish: try to add any remaining edges
    if best_result is not None:
        final_chosen = set(tuple(e) for e in best_result)
        for edge in all_edges:
            x, y, bit_i = edge
            if (x, y) not in final_chosen and is_valid_add(final_chosen, edge):
                best_result.append([x, y])
                final_chosen.add((x, y))

    if best_result is None:
        # Fallback to deterministic greedy
        chosen = set()
        best_result = []
        for edge in all_edges:
            x, y, bit_i = edge
            if is_valid_add(chosen, edge):
                chosen.add((x, y))
                best_result.append([x, y])

    return best_result
# EVOLVE-BLOCK-END
