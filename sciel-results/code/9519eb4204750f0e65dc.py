import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s limit

    # --- Core scoring helpers ---
    def _doubled_areas(points):
        n = len(points)
        areas = []
        for i in range(n):
            ax, ay = points[i]
            for j in range(i+1, n):
                bx, by = points[j]
                dx1 = bx - ax
                dy1 = by - ay
                for k in range(j+1, n):
                    cx, cy = points[k]
                    dx2 = cx - ax
                    dy2 = cy - ay
                    det = dx1 * dy2 - dy1 * dx2
                    if det < 0:
                        det = -det
                    if det > 0:
                        areas.append(det)
        return areas

    def _area_counter(points):
        return Counter(_doubled_areas(points))

    def _max_multiplicity(points):
        areas = _doubled_areas(points)
        if not areas:
            return 0
        return Counter(areas).most_common(1)[0][1]

    def _score(points):
        if len(points) != 30:
            return 0
        if len(set(points)) != 30:
            return 0
        for x, y in points:
            if x < -30 or x > 30 or y < -30 or y > 30:
                return 0
        return _max_multiplicity(points)

    def _fast_score(points, prev_counter, moved_idx, old_point, new_point):
        counter = Counter(prev_counter)
        n = len(points)
        idx = moved_idx

        # Remove old contributions involving old_point at idx
        ox, oy = old_point
        for i in range(n):
            if i == idx:
                continue
            px, py = points[i]
            for j in range(i+1, n):
                if j == idx:
                    continue
                qx, qy = points[j]
                det = (px - ox) * (qy - oy) - (py - oy) * (qx - ox)
                if det < 0:
                    det = -det
                if det > 0:
                    counter[det] -= 1
                    if counter[det] <= 0:
                        del counter[det]

        # Add new contributions involving new_point at idx
        nx, ny = new_point
        for i in range(n):
            if i == idx:
                continue
            px, py = points[i]
            for j in range(i+1, n):
                if j == idx:
                    continue
                qx, qy = points[j]
                det = (px - nx) * (qy - ny) - (py - ny) * (qx - nx)
                if det < 0:
                    det = -det
                if det > 0:
                    counter[det] += 1

        if not counter:
            return 0, counter
        return max(counter.values()), counter

    # --- Structured initial candidates ---
    def _structured_candidates():
        candidates = []
        # Standard grids m*n=30
        for m, n in [(6,5), (5,6), (10,3), (3,10), (15,2), (2,15)]:
            pts = []
            for i in range(m):
                for j in range(n):
                    x = i - m // 2
                    y = j - n // 2
                    pts.append((x, y))
            if len(pts) == 30 and len(set(pts)) == 30:
                if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
                    candidates.append(pts)

        # Grids with spacing
        for m, n in [(6,5), (5,6), (10,3)]:
            for sx in [1, 2, 3]:
                for sy in [1, 2, 3]:
                    pts = []
                    for i in range(m):
                        for j in range(n):
                            x = i * sx - (m * sx) // 2
                            y = j * sy - (n * sy) // 2
                            pts.append((x, y))
                    if len(pts) == 30 and len(set(pts)) == 30:
                        if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
                            candidates.append(pts)

        # Staggered rows
        for dx in [1, 2, 3]:
            for dy in [1, 2, 3]:
                pts = []
                for row in range(5):
                    for col in range(6):
                        x = col * dx + (row % 2) * (dx // 2 + 1) - 15
                        y = row * dy - 10
                        pts.append((x, y))
                if len(pts) == 30 and len(set(pts)) == 30:
                    if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
                        candidates.append(pts)

        return candidates

    # --- Local search with swaps and moves ---
    def _local_search(points, time_limit):
        best_points = list(points)
        best_score = _score(best_points)
        best_counter = _area_counter(best_points)
        occupied = set(best_points)

        # Neighborhoods for single-point moves
        small_nh = [(dx, dy) for dx in range(-4, 5) for dy in range(-4, 5) if (dx, dy) != (0, 0)]
        med_nh = [(dx, dy) for dx in range(-8, 9) for dy in range(-8, 9) if (dx, dy) != (0, 0)]
        large_nh = [(dx, dy) for dx in range(-15, 16) for dy in range(-15, 16) if (dx, dy) != (0, 0)]
        neighborhoods = [small_nh, med_nh, large_nh]

        iteration = 0
        while time.time() < time_limit:
            # Cycle neighborhood size
            nh_size = iteration % 3
            if time.time() > time_limit - 15:
                nh_size = 0  # Use small neighborhood near deadline

            improved_this_pass = False

            # Single-point moves
            for idx in range(30):
                if time.time() > time_limit:
                    break
                px, py = best_points[idx]
                for dx, dy in neighborhoods[nh_size]:
                    if time.time() > time_limit:
                        break
                    nx, ny = px + dx, py + dy
                    if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                        continue
                    new_point = (nx, ny)
                    if new_point in occupied:
                        continue

                    new_points = list(best_points)
                    new_points[idx] = new_point
                    s, new_counter = _fast_score(new_points, best_counter, idx, (px, py), new_point)

                    if s > best_score:
                        # Update occupied set
                        occupied.remove((px, py))
                        occupied.add(new_point)
                        best_score = s
                        best_points = new_points
                        best_counter = new_counter
                        improved_this_pass = True
                        break  # Restart scan with improved state

            if time.time() > time_limit:
                break

            # Swap operations: exchange two points' positions
            for i in range(30):
                if time.time() > time_limit:
                    break
                pi = best_points[i]
                for j in range(i+1, 30):
                    if time.time() > time_limit:
                        break
                    pj = best_points[j]

                    # Try swapping: put pj at i's position and pi at j's position
                    # Actually, swap means point i goes to where j was and vice versa
                    # But that's just reordering, which doesn't change the set.
                    # Wait, swap in the context of this problem usually means
                    # moving point i to a new location and point j to another.
                    # Let's interpret "swap" as: pick two points, move each to the other's position? No, that's identity.
                    # Let's interpret "swap" as: pick two points i, j. Move i to a random nearby spot, j to a random nearby spot.
                    # Or better: pick two points i, j. Try moving i to j's location and j to i's location? That's just swapping indices, no change to set.
                    # The inspiration says "swaps allow reconfiguring pairs".
                    # Let's try: for each pair (i,j), try moving i to a new location and j to a new location simultaneously.
                    # This is expensive. Let's limit this.
                    
                    # Alternative swap: move point i to point j's location, and point j to a new location?
                    # Let's stick to a simpler "pair move": pick i, j. Try moving i to a new spot (checking collision with j's new spot?).
                    # Let's implement a simple pair move: pick i, j. Move i to a random nearby spot. Then move j to a random nearby spot.
                    # This is complex to score incrementally.
                    
                    # Let's use a different interpretation: "swap" might mean exchanging the coordinates of two points? No.
                    # Let's try: pick i, j. Try the move where point i goes to (px + dx, py + dy) and point j goes to (qx + dx', qy + dy').
                    # Too many combinations.
                    
                    # Let's try a simpler heuristic: occasionally, pick two points and try to move both to new random positions.
                    if iteration % 10 == 0:  # Do pair moves less frequently
                        # Pick two random indices
                        # Already have i, j
                        # Try moving i to a new spot and j to a new spot
                        # For efficiency, only try a few random moves per pair
                        for _ in range(2):
                            if time.time() > time_limit:
                                break
                            new_pi = (pi[0] + random.randint(-5, 5), pi[1] + random.randint(-5, 5))
                            new_pj = (pj[0] + random.randint(-5, 5), pj[1] + random.randint(-5, 5))
                            if new_pi[0] < -30 or new_pi[0] > 30 or new_pi[1] < -30 or new_pi[1] > 30:
                                continue
                            if new_pj[0] < -30 or new_pj[0] > 30 or new_pj[1] < -30 or new_pj[1] > 30:
                                continue
                            if new_pi in occupied or new_pj in occupied:
                                continue
                            if new_pi == new_pj:
                                continue
                            # Check if new_pi or new_pj collide with existing points (excluding i and j)
                            # occupied contains all points. We need to remove i and j first to check collisions.
                            # This is getting complicated. Let's skip complex pair moves for now and rely on single moves with larger neighborhoods.
                            pass

            iteration += 1
            if not improved_this_pass and iteration % 50 == 0:
                # If stuck, try a random restart within the current best's neighborhood
                # Or just continue with larger neighborhood
                pass

        return best_points, best_score

    # --- Main logic ---
    # Generate initial candidates and pick the best
    candidates = _structured_candidates()
    if not candidates:
        # Fallback: 6x5 grid
        candidates = [[(x, y) for x in range(6) for y in range(5)]]

    best_initial = None
    best_initial_score = 0
    for cand in candidates:
        s = _score(cand)
        if s > best_initial_score:
            best_initial_score = s
            best_initial = cand

    if best_initial is None:
        best_initial = [(x, y) for x in range(6) for y in range(5)]
        best_initial_score = _score(best_initial)

    # Run local search
    best_points, best_score = _local_search(best_initial, deadline)

    # If we have time, try a few more restarts from slightly perturbed best_initial
    if time.time() < deadline - 10:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            # Perturb best_initial slightly
            perturbed = list(best_initial)
            idx = random.randint(0, 29)
            px, py = perturbed[idx]
            new_x = px + random.randint(-3, 3)
            new_y = py + random.randint(-3, 3)
            if -30 <= new_x <= 30 and -30 <= new_y <= 30:
                new_pt = (new_x, new_y)
                if new_pt not in perturbed:
                    perturbed[idx] = new_pt
            s = _score(perturbed)
            if s > best_score:
                # Run local search from this perturbed state
                pt, sc = _local_search(perturbed, deadline)
                if sc > best_score:
                    best_points = pt
                    best_score = sc

    return best_points
