import time
import random

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute neighbor masks for all vertices
# NEIGHBOR_MASKS[v] is a bitmask where bit i is set if vertex i is a neighbor of v
NEIGHBOR_MASKS = [0] * SIZE
for v in range(SIZE):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ (1 << b))
    NEIGHBOR_MASKS[v] = m

def construct():
    deadline = time.time() + 115.0
    best_path = [0]
    best_len = 1
    
    def backtrack_improve(path):
        """
        Try to extend the path by backtracking.
        Returns the improved path.
        """
        def build_masks(p):
            used = 0
            forbidden = 0
            for i, v in enumerate(p):
                used |= 1 << v
                if i < len(p) - 1:
                    forbidden |= NEIGHBOR_MASKS[v]
            return used, forbidden

        used, forbidden = build_masks(path)
        p = list(path)
        
        def limited_dfs(cur_path, cur_used, cur_forbidden, max_depth):
            if max_depth == 0:
                return cur_path
            
            cur = cur_path[-1]
            candidates = NEIGHBOR_MASKS[cur] & ~(cur_used | cur_forbidden)
            
            if candidates == 0:
                return cur_path
            
            best = cur_path
            tmp = candidates
            
            while tmp:
                lsb = tmp & (-tmp)
                v = lsb.bit_length() - 1
                tmp ^= lsb
                
                new_path = cur_path + [v]
                new_used = cur_used | (1 << v)
                new_forbidden = cur_forbidden | NEIGHBOR_MASKS[cur]
                
                ext = limited_dfs(new_path, new_used, new_forbidden, max_depth - 1)
                if len(ext) > len(best):
                    best = ext
            
            return best

        # Try to improve with depth 4
        improved_path = limited_dfs(p, used, forbidden, 4)
        
        if len(improved_path) > len(p):
            return improved_path
        return p

    def beam_search(start, width, rng):
        """
        Beam search using bitmask state.
        """
        beam = [([start], 1 << start, 0)]
        best_local_path = [start]
        best_local_len = 1
        
        while True:
            if time.time() > deadline - 2:
                break
            
            new_beam = []
            seen_suffixes = set()
            
            for path, used_mask, forbidden_mask in beam:
                cur = path[-1]
                candidates = NEIGHBOR_MASKS[cur] & ~(used_mask | forbidden_mask)
                
                if candidates == 0:
                    if len(path) > best_local_len:
                        best_local_len = len(path)
                        best_local_path = list(path)
                    continue
                
                tmp = candidates
                while tmp:
                    lsb = tmp & (-tmp)
                    v = lsb.bit_length() - 1
                    tmp ^= lsb
                    
                    new_path = path + [v]
                    new_used = used_mask | (1 << v)
                    new_forbidden = forbidden_mask | NEIGHBOR_MASKS[cur]
                    
                    key = tuple(new_path[-4:])
                    if key in seen_suffixes:
                        continue
                    seen_suffixes.add(key)
                    
                    new_beam.append((new_path, new_used, new_forbidden))
                    
                    if len(new_path) > best_local_len:
                        best_local_len = len(new_path)
                        best_local_path = list(new_path)
            
            if not new_beam:
                break
            
            new_beam.sort(key=lambda x: -len(x[0]))
            beam = new_beam[:width]
            
        return best_local_path, best_local_len

    rng = random.Random(42)
    
    widths = [50, 100, 200, 500]
    
    while time.time() < deadline - 5:
        width = rng.choice(widths)
        start = rng.randint(0, SIZE - 1)
        
        p, L = beam_search(start, width, rng)
        
        # Try to improve
        p_improved = backtrack_improve(p)
        
        if len(p_improved) > best_len:
            best_len = len(p_improved)
            best_path = p_improved
            
    return best_path

# Test locally
if __name__ == "__main__":
    res = construct()
    print(len(res))
