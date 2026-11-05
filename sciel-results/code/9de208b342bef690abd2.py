<<<<<<< SEARCH
"""Rewritten program for minimum maximal no-isosceles set on [0,16)^2."""
N = 16
import random
import time
import itertools
import math

def construct():
    """
    Construct a maximal no-isosceles set (no three points form an isosceles triangle)
    that is minimal in size. Uses greedy construction followed by local search
    (swap/replace moves) and random restarts to escape local minima.
    """
    deadline = time.time() + 280  # safety margin before 330s kill
    
    def is_iso(a, b, c):
        ab2 = (a[0]-b[0])**2 + (a[1]-b[1])**2
        ac2 = (a[0]-c[0])**2 + (a[1]-c[1])**2
        bc2 = (b[0]-c[0])**2 + (b[1]-c[1])**2
        return ab2 == ac2 or ab2 == bc2 or ac2 == bc2
    
    def is_valid_set(points):
        """Check that no three points in the set form an isosceles triangle."""
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_iso(points[i], points[j], points[k]):
                        return False
        return True
    
    def is_maximal(points):
        """Check that no grid point can be added without creating an isosceles triangle."""
        point_set = set(points)
        for p in itertools.product(range(N), repeat=2):
            if p in point_set:
                continue
            # Check if adding p creates any isosceles triangle
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    if is_iso(p, points[i], points[j]):
                        break
                else:
                    continue
                break
            else:
                return False  # p can be added, so not maximal
        return True
    
    def greedy_construct(order):
        """Greedy construction following the given order of points."""
        chosen = []
        chosen_set = set()
        for p in order:
            if p in chosen_set:
                continue
            ok = True
            for i in range(len(chosen)):
                for j in range(i+1, len(chosen)):
                    if is_iso(p, chosen[i], chosen[j]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                chosen.append(p)
                chosen_set.add(p)
        return chosen
    
    def try_add_point(points, p):
        """Try to add point p to the set, returning new set if valid."""
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                if is_iso(p, points[i], points[j]):
                    return None
        return points + [p]
    
    def local_search(points):
        """
        Local search: try to remove points while maintaining maximality,
        then try to re-add different points. Also try swap moves.
        """
        if time.time() > deadline - 10:
            return points
        
        points = list(points)
        improved = True
        max_iters = 500
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline - 15:
            improved = False
            iters += 1
            n = len(points)
            
            # Try removing each point
            for i in range(n):
                candidate = points[:i] + points[i+1:]
                if is_valid_set(candidate):
                    # Check if still maximal
                    if is_maximal(candidate):
                        points = candidate
                        improved = True
                        break
            if improved:
                continue
            
            # Try swap: remove one point, add another
            for i in range(len(points)):
                remaining = points[:i] + points[i+1:]
                for p in itertools.product(range(N), repeat=2):
                    if p in remaining:
                        continue
                    new_set = remaining + [p]
                    if is_valid_set(new_set) and is_maximal(new_set):
                        if len(new_set) < len(points):
                            points = new_set
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue
            
            # Try adding a point (if not maximal, make it maximal)
            if not is_maximal(points):
                # Find a point that can be added
                for p in itertools.product(range(N), repeat=2):
                    if p in points:
                        continue
                    new_set = try_add_point(points, p)
                    if new_set is not None:
                        points = new_set
                        improved = True
                        break
        
        return points
    
    best = None
    best_len = float('inf')
    
    rng = random.Random(42)
    
    while time.time() < deadline - 30:
        # Generate a random order for greedy construction
        all_points = list(itertools.product(range(N), repeat=2))
        rng.shuffle(all_points)
        
        # Greedy construction
        points = greedy_construct(all_points)
        
        # Ensure maximality by adding points
        if not is_maximal(points):
            # Add points until maximal
            changed = True
            while changed:
                changed = False
                for p in all_points:
                    if p in points:
                        continue
                    new_set = try_add_point(points, p)
                    if new_set is not None:
                        points = new_set
                        changed = True
                        break
        
        # Local search to minimize
        points = local_search(points)
        
        if len(points) < best_len:
            best = points
            best_len = len(points)
        
        # Early termination if we find a very good solution
        if best_len <= 10:
            break
    
    if best is None:
        # Fallback: simple greedy
        best = greedy_construct(list(itertools.product(range(N), repeat=2)))
        # Ensure maximal
        if not is_maximal(best):
            for p in itertools.product(range(N), repeat=2):
                if p in best:
                    continue
                new_set = try_add_point(best, p)
                if new_set is not None:
                    best = new_set
    
    return best
=======
"""Rewritten program for minimum maximal no-isosceles set on [0,16)^2."""
N = 16
TOTAL = N * N
import random
import time
import itertools
import math

# Precompute isosceles conflicts: for each pair (i,j), which points k form an isosceles triangle with i,j
def _precompute_conflicts():
    """
    For each point index p, compute a bitmask of points q such that
    there exists some point r where {p, q, r} forms an isosceles triangle.
    This allows O(1) check: can we add p to set S? Yes iff (conflict_mask[p] & S_mask) == 0.
    """
    points = list(itertools.product(range(N), repeat=2))
    idx = {p: i for i, p in enumerate(points)}
    
    # conflict_mask[p] = bitmask of all q such that p and q are in some isosceles triple
    conflict_mask = [0] * TOTAL
    
    # For each pair (i, j), find all k such that {i, j, k} is isosceles
    for i in range(TOTAL):
        for j in range(i + 1, TOTAL):
            pi, pj = points[i], points[j]
            ij2 = (pi[0]-pj[0])**2 + (pi[1]-pj[1])**2
            
            for k in range(TOTAL):
                if k == i or k == j:
                    continue
                pk = points[k]
                ik2 = (pi[0]-pk[0])**2 + (pi[1]-pk[1])**2
                jk2 = (pj[0]-pk[0])**2 + (pj[1]-pk[1])**2
                if ik2 == ij2 or ik2 == jk2 or ij2 == jk2:
                    # i, j, k form isosceles triangle
                    conflict_mask[i] |= (1 << j)
                    conflict_mask[j] |= (1 << i)
                    conflict_mask[i] |= (1 << k)
                    conflict_mask[k] |= (1 << i)
                    conflict_mask[j] |= (1 << k)
                    conflict_mask[k] |= (1 << j)
    
    return points, conflict_mask

# Precompute at module level
POINTS, CONFLICT_MASK = _precompute_conflicts()
POINT_INDEX = {p: i for i, p in enumerate(POINTS)}

def construct():
    """
    Construct a maximal no-isosceles set (no three points form an isosceles triangle)
    that is minimal in size. Uses precomputed conflict bitmasks for fast validity checks,
    greedy construction, and local search with random restarts.
    """
    deadline = time.time() + 280  # safety margin before 330s kill
    
    def is_valid_mask(mask):
        """Check if the set represented by mask has no isosceles triangles.
        For each point p in mask, no other point q in mask should be in conflict_mask[p]."""
        m = mask
        while m:
            lsb = m & (-m)
            p = lsb.bit_length() - 1
            if CONFLICT_MASK[p] & m:
                return False
            m ^= lsb
        return True
    
    def is_maximal_mask(mask):
        """Check that no grid point can be added without creating an isosceles triangle."""
        for p in range(TOTAL):
            if mask & (1 << p):
                continue
            # Can we add p? Check if p conflicts with any point in mask
            if CONFLICT_MASK[p] & mask == 0:
                return False  # p can be added, so not maximal
        return True
    
    def greedy_construct(order):
        """Greedy construction following the given order of points. Returns mask."""
        mask = 0
        for p in order:
            idx = POINT_INDEX[p]
            if CONFLICT_MASK[idx] & mask == 0:
                mask |= (1 << idx)
        return mask
    
    def make_maximal(mask):
        """Add points until the set is maximal."""
        changed = True
        while changed:
            changed = False
            for p in range(TOTAL):
                if mask & (1 << p):
                    continue
                if CONFLICT_MASK[p] & mask == 0:
                    mask |= (1 << p)
                    changed = True
        return mask
    
    def local_search(mask):
        """
        Local search: try to remove points while maintaining maximality.
        Uses bitmask operations for speed.
        """
        if time.time() > deadline - 10:
            return mask
        
        mask = mask
        improved = True
        max_iters = 200
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline - 15:
            improved = False
            iters += 1
            n = mask.bit_count()
            
            # Try removing each point
            for i in range(n):
                # Find the i-th set bit
                tmp = mask
                for _ in range(i):
                    tmp &= tmp - 1
                lsb = tmp & (-tmp)
                p = lsb.bit_length() - 1
                
                candidate = mask & ~(1 << p)
                if is_maximal_mask(candidate):
                    mask = candidate
                    improved = True
                    break
            
            if not improved:
                # Try swap: remove one point, add another
                for i in range(n):
                    tmp = mask
                    for _ in range(i):
                        tmp &= tmp - 1
                    lsb = tmp & (-tmp)
                    p = lsb.bit_length() - 1
                    
                    remaining = mask & ~(1 << p)
                    # Try adding each point not in remaining
                    for q in range(TOTAL):
                        if remaining & (1 << q):
                            continue
                        if CONFLICT_MASK[q] & remaining == 0:
                            new_mask = remaining | (1 << q)
                            if is_maximal_mask(new_mask):
                                if new_mask.bit_count() < n:
                                    mask = new_mask
                                    improved = True
                                    break
                    if improved:
                        break
            
            if not improved and not is_maximal_mask(mask):
                # Make it maximal
                mask = make_maximal(mask)
        
        return mask
    
    def mask_to_points(mask):
        """Convert mask to list of (x,y) points."""
        result = []
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            result.append(POINTS[idx])
            m ^= lsb
        return result
    
    best = None
    best_len = float('inf')
    
    rng = random.Random(42)
    
    while time.time() < deadline - 30:
        # Generate a random order for greedy construction
        order = list(range(TOTAL))
        rng.shuffle(order)
        
        # Greedy construction
        mask = greedy_construct(order)
        
        # Ensure maximality
        mask = make_maximal(mask)
        
        # Local search to minimize
        mask = local_search(mask)
        
        if mask.bit_count() < best_len:
            best = mask
            best_len = mask.bit_count()
        
        # Early termination if we find a very good solution
        if best_len <= 10:
            break
    
    if best is None:
        # Fallback: simple greedy
        order = list(range(TOTAL))
        best = greedy_construct(order)
        best = make_maximal(best)
    
    return mask_to_points(best)
>>>>>>> REPLACE
