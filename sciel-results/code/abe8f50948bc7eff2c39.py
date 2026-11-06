"""Snake-in-the-box in Q_9: construct an induced path maximizing its length."""
import time
import random
import sys

N = 9
MASK = (1 << N) - 1
DEADLINE = 145.0  # safety margin below 150s

def construct():
    start = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start)
    
    # Precompute neighbors for all vertices
    # neighbors[v] = list of 9 neighbors of v
    neighbors = []
    for v in range(1 << N):
        nbrs = []
        for b in range(N):
            nbrs.append(v ^ (1 << b))
        neighbors.append(nbrs)
    
    # Precompute bit_count for XOR differences
    # We'll use int.bit_count() which is fast in Python 3.10+
    
    def is_valid_extension(path, used, candidate):
        """Check if candidate can be added to path maintaining induced property."""
        cur = path[-1]
        # Must differ in exactly one bit from cur
        if (candidate ^ cur).bit_count() != 1:
            return False
        # Must not be in used
        if candidate in used:
            return False
        # Must not be at distance 1 from any non-adjacent vertex in path
        # Non-adjacent means not path[-1] (which is cur)
        for i in range(len(path) - 1):
            if (candidate ^ path[i]).bit_count() == 1:
                return False
        return True
    
    def greedy_extend(path, used, bit_order=None):
        """Extend path greedily. Returns new path."""
        if bit_order is None:
            bit_order = list(range(N))
        p = list(path)
        u = set(used)
        while True:
            cur = p[-1]
            # Get candidates sorted by bit_order
            best_candidate = None
            best_score = -1
            for b in bit_order:
                v = cur ^ (1 << b)
                if v in u:
                    continue
                if (v ^ cur).bit_count() != 1:
                    continue
                # Check induced property
                ok = True
                for i in range(len(p) - 1):
                    if (v ^ p[i]).bit_count() == 1:
                        ok = False
                        break
                if ok:
                    # Score: prefer vertices with more available future options
                    # Count how many of v's neighbors are still available
                    future_opts = 0
                    for nb in neighbors[v]:
                        if nb not in u and nb != cur:
                            future_opts += 1
                    if future_opts > best_score:
                        best_score = future_opts
                        best_candidate = v
            if best_candidate is None:
                return p
            p.append(best_candidate)
            u.add(best_candidate)
    
    def warnsdorff_search(path, used, max_depth=0):
        """
        Backtracking search with Warnsdorff's rule:
        At each step, try the candidate with the fewest future options first.
        This helps avoid dead-ends.
        """
        if time_left() < 1.0:
            return list(path)
        
        cur = path[-1]
        # Find all valid candidates
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            if (v ^ cur).bit_count() != 1:
                continue
            # Check induced property
            ok = True
            for i in range(len(path) - 1):
                if (v ^ path[i]).bit_count() == 1:
                    ok = False
                    break
            if ok:
                # Count future options for v
                future_opts = 0
                for nb in neighbors[v]:
                    if nb not in used and nb != cur:
                        future_opts += 1
                candidates.append((future_opts, v))
        
        if not candidates:
            return list(path)
        
        # Sort by fewest future options first (Warnsdorff)
        candidates.sort()
        
        best = list(path)
        
        for _, v in candidates:
            if time_left() < 1.0:
                break
            new_path = path + [v]
            new_used = used | {v}
            if max_depth > 0:
                result = warnsdorff_search(new_path, new_used, max_depth - 1)
            else:
                # Continue greedily from here
                result = greedy_extend(new_path, new_used)
            if len(result) > len(best):
                best = result
        
        return best
    
    def backtrack_with_lookahead(path, used, depth, max_depth):
        """
        Deeper backtracking with lookahead.
        """
        if depth >= max_depth:
            return greedy_extend(path, used)
        
        cur = path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            if (v ^ cur).bit_count() != 1:
                continue
            ok = True
            for i in range(len(path) - 1):
                if (v ^ path[i]).bit_count() == 1:
                    ok = False
                    break
            if ok:
                # Lookahead: count options after adding v
                future_opts = 0
                for nb in neighbors[v]:
                    if nb not in used and nb != cur:
                        # Check if nb would be valid
                        nb_ok = True
                        for j in range(len(path) - 1):
                            if (nb ^ path[j]).bit_count() == 1:
                                nb_ok = False
                                break
                        if nb_ok:
                            future_opts += 1
                candidates.append((future_opts, v))
        
        if not candidates:
            return list(path)
        
        candidates.sort()
        best = list(path)
        
        for _, v in candidates:
            if time_left() < 1.0:
                break
            new_path = path + [v]
            new_used = used | {v}
            result = backtrack_with_lookahead(new_path, new_used, depth + 1, max_depth)
            if len(result) > len(best):
                best = result
        
        return best
    
    def splice_improve(path):
        """
        Try to improve by finding alternative extensions at various points.
        For each position in the path, try different next vertices and see if
        it leads to a longer total path.
        """
        best = list(path)
        n = len(path)
        
        # Try extending from the end with different strategies
        for trial in range(20):
            if time_left() < 2.0:
                break
            # Random bit order for greedy
            order = list(range(N))
            random.shuffle(order)
            ext = greedy_extend(path, set(path), order)
            if len(ext) > len(best):
                best = ext
        
        # Try splicing: at position i, replace suffix with a better one
        for i in range(max(1, n - 30), n - 5):
            if time_left() < 2.0:
                break
            prefix = path[:i+1]
            used_prefix = set(prefix)
            # Try to extend from prefix with different orders
            for trial in range(5):
                if time_left() < 1.0:
                    break
                order = list(range(N))
                random.shuffle(order)
                ext = greedy_extend(prefix, used_prefix, order)
                if len(ext) > len(best):
                    best = ext
        
        return best
    
    # Phase 1: Get initial snake with randomized greedy + Warnsdorff
    best_path = []
    for trial in range(300):
        if time_left() < 10.0:
            break
        order = list(range(N))
        random.shuffle(order)
        p = [0]
        u = {0}
        while True:
            cur = p[-1]
            # Find best candidate using Warnsdorff
            candidates = []
            for b in order:
                v = cur ^ (1 << b)
                if v in u:
                    continue
                if (v ^ cur).bit_count() != 1:
                    continue
                ok = True
                for i in range(len(p) - 1):
                    if (v ^ p[i]).bit_count() == 1:
                        ok = False
                        break
                if ok:
                    # Count future options
                    future_opts = 0
                    for nb in neighbors[v]:
                        if nb not in u and nb != cur:
                            future_opts += 1
                    candidates.append((future_opts, v))
            
            if not candidates:
                break
            
            # Pick the one with fewest future options (Warnsdorff)
            candidates.sort()
            _, nxt = candidates[0]
            p.append(nxt)
            u.add(nxt)
        
        if len(p) > len(best_path):
            best_path = p
    
    # Phase 2: Improve best_path with splicing and different extensions
    if time_left() > 10.0:
        improved = splice_improve(best_path)
        if len(improved) > len(best_path):
            best_path = improved
    
    # Phase 3: Deeper backtracking from current best
    if time_left() > 15.0:
        # Try Warnsdorff backtracking from the end
        max_depth = min(25, int(time_left() * 0.5))
        result = warnsdorff_search(best_path, set(best_path), max_depth)
        if len(result) > len(best_path):
            best_path = result
    
    # Phase 4: More random restarts with improved greedy
    while time_left() > 5.0:
        order = list(range(N))
        random.shuffle(order)
        p = [0]
        u = {0}
        while True:
            cur = p[-1]
            candidates = []
            for b in order:
                v = cur ^ (1 << b)
                if v in u:
                    continue
                if (v ^ cur).bit_count() != 1:
                    continue
                ok = True
                for i in range(len(p) - 1):
                    if (v ^ p[i]).bit_count() == 1:
                        ok = False
                        break
                if ok:
                    future_opts = 0
                    for nb in neighbors[v]:
                        if nb not in u and nb != cur:
                            future_opts += 1
                    candidates.append((future_opts, v))
            
            if not candidates:
                break
            
            candidates.sort()
            _, nxt = candidates[0]
            p.append(nxt)
            u.add(nxt)
        
        if len(p) > len(best_path):
            best_path = p
            # Try to extend further
            if time_left() > 10.0:
                max_depth = min(20, int(time_left() * 0.3))
                result = warnsdorff_search(best_path, set(best_path), max_depth)
                if len(result) > len(best_path):
                    best_path = result
    
    # Phase 5: Final improvement attempts
    if time_left() > 5.0:
        improved = splice_improve(best_path)
        if len(improved) > len(best_path):
            best_path = improved
    
    return best_path

if __name__ == "__main__":
    result = construct()
    print(f"Length: {len(result) - 1}")
