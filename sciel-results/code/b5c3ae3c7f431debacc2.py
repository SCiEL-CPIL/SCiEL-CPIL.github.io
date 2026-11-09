import time
from bisect import bisect_left

_P = 1009
_M = 5


def _build_residue_list():
    # Returns a sorted list of non-zero quadratic residues in [1, 1008]
    residues = set()
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0:
            residues.add(r)
    return sorted(residues)


def _build_root_map(residue_list):
    # For each residue in the sorted list, store one root
    # residue_list[i] -> r_i, we want r_i^2 % P == residue_list[i]
    # We can re-derive or store during precomputation
    root_map = {}
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0 and r not in root_map:
            root_map[r] = x
    return root_map


def construct():
    start_time = time.monotonic()
    deadline = start_time + 55.0

    # Precompute once
    residue_list = _build_residue_list()
    root_map = _build_root_map(residue_list)

    # For a run of length m=5 starting at s, we need:
    # s, s+1, s+2, s+3, s+4 all in residue set
    # Equivalently, for each residue r in residue_list, if r >= 4,
    # we can check if r, r+1, r+2, r+3, r+4 are all in the set.
    # But s must be in [1, 1004], so r+4 <= 1008 => r <= 1004.
    
    # Since we want the MINIMAL s, and s = r (where r is the first element of the run),
    # we just need to find the smallest r in residue_list such that
    # r, r+1, r+2, r+3, r+4 are all in residue_list.
    
    # Use binary search for O(log n) lookups
    for r in residue_list:
        if time.monotonic() > deadline:
            # Return best found so far if we've already checked some? 
            # Actually we iterate in sorted order, so if we haven't found one yet,
            # we should raise or return None. But the problem guarantees a solution.
            raise RuntimeError("Deadline exceeded")
            
        # Check if r+1, r+2, r+3, r+4 are in the set
        # We can use bisect to check membership in the sorted list
        found = True
        for i in range(1, _M):
            target = r + i
            if target > _P - 1:
                found = False
                break
            idx = bisect_left(residue_list, target)
            if idx >= len(residue_list) or residue_list[idx] != target:
                found = False
                break
        
        if found:
            s = r
            roots = [root_map[s + i] for i in range(_M)]
            return {'start': s, 'roots': roots}

    raise RuntimeError("No run found")
