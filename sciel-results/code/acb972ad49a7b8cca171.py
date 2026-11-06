import time
import random
import sys

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # Safety margin for 60s limit

def construct():
    start_time = time.time()
    best_size = 0
    best_A = []
    best_B = []
    
    def is_valid_add(x, elems, sum_bits, diff_bits, other_diff_bits):
        """Check if x can be added to the set."""
        if (sum_bits >> (x + x)) & 1:
            return False
        for a in elems:
            s = x + a
            if (sum_bits >> s) & 1:
                return False
            d = abs(x - a)
            if (diff_bits >> d) & 1:
                return False
            if (other_diff_bits >> d) & 1:
                return False
        return True

    def add_element(x, elems, sum_bits, diff_bits):
        """Add x to set, return updated (elems, sum_bits, diff_bits)."""
        new_sum_bits = sum_bits | (1 << (x + x))
        new_diff_bits = diff_bits
        for a in elems:
            new_sum_bits |= (1 << (x + a))
            d = abs(x - a)
            new_diff_bits |= (1 << d)
        return elems + [x], new_sum_bits, new_diff_bits

    def remove_element(x, elems, sum_bits, diff_bits):
        """Remove x from set, return updated (elems, sum_bits, diff_bits)."""
        # Since A is Sidon, all sums and non-zero diffs are unique.
        # We can just clear the bits.
        new_sum_bits = sum_bits & ~(1 << (x + x))
        new_diff_bits = diff_bits
        new_elems = []
        for a in elems:
            if a == x:
                continue
            new_elems.append(a)
            s = x + a
            new_sum_bits &= ~(1 << s)
            d = abs(x - a)
            new_diff_bits &= ~(1 << d)
        return new_elems, new_sum_bits, new_diff_bits

    def greedy_construct(order=None):
        """Greedy construction with optional custom order."""
        A = []
        B = []
        sumA = 0
        diffA = 0
        sumB = 0
        diffB = 0
        
        if order is None:
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
        else:
            candidates = order
        
        for x in candidates:
            if len(A) < MAX_EACH and is_valid_add(x, A, sumA, diffA, diffB):
                A, sumA, diffA = add_element(x, A, sumA, diffA)
                continue
            if len(B) < MAX_EACH and is_valid_add(x, B, sumB, diffB, diffA):
                B, sumB, diffB = add_element(x, B, sumB, diffB)
                continue
        
        return A, B, sumA, diffA, sumB, diffB

    def local_search(A, B, sumA, diffA, sumB, diffB):
        """1-for-2 swap local search with incremental updates."""
        if time.time() - start_time > DEADLINE - 8:
            return A, B, sumA, diffA, sumB, diffB
        
        used = set(A) | set(B)
        
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Try removing from A
            for i in range(len(A)):
                if time.time() - start_time > DEADLINE - 5:
                    break
                a = A[i]
                new_A, new_sumA, new_diffA = remove_element(a, A, sumA, diffA)
                
                # Find all candidates valid for new_A
                # Note: we must check against new_diffA and diffB
                valid_adds_A = []
                for x in range(1, N + 1):
                    if x in used and x != a:
                        continue
                    if len(new_A) < MAX_EACH and is_valid_add(x, new_A, new_sumA, new_diffA, diffB):
                        valid_adds_A.append(x)
                
                # Try 1-for-2: add two to A
                # Optimization: Instead of rebuilding for every pair, we can check pairs more efficiently
                # But let's stick to the logic, just faster.
                found_swap = False
                for j in range(len(valid_adds_A)):
                    x1 = valid_adds_A[j]
                    # Temporarily add x1 to new_A to check x2
                    # We can simulate this
                    temp_sumA = new_sumA | (1 << (x1 + x1))
                    temp_diffA = new_diffA
                    for y in new_A:
                        temp_sumA |= (1 << (x1 + y))
                        temp_diffA |= (1 << abs(x1 - y))
                    
                    for k in range(j + 1, len(valid_adds_A)):
                        x2 = valid_adds_A[k]
                        # Check if x2 is valid with x1 added
                        if (temp_sumA >> (x2 + x2)) & 1: continue
                        conflict = False
                        for y in new_A:
                            if (temp_sumA >> (x2 + y)) & 1:
                                conflict = True; break
                            d = abs(x2 - y)
                            if (temp_diffA >> d) & 1:
                                conflict = True; break
                            if (diffB >> d) & 1:
                                conflict = True; break
                        if not conflict:
                            # Check x1 vs x2
                            if (temp_sumA >> (x1 + x2)) & 1: conflict = True
                            else:
                                d12 = abs(x1 - x2)
                                if (temp_diffA >> d12) & 1: conflict = True
                                elif (diffB >> d12) & 1: conflict = True
                            
                            if not conflict:
                                A = new_A + [x1, x2]
                                # Update bitmasks
                                sumA = temp_sumA | (1 << (x2 + x2))
                                diffA = temp_diffA
                                for y in new_A:
                                    sumA |= (1 << (x2 + y))
                                    diffA |= (1 << abs(x2 - y))
                                diffA |= (1 << abs(x1 - x2))
                                
                                B, sumB, diffB = B, sumB, diffB # No change
                                used = set(A) | set(B)
                                improved = True
                                found_swap = True
                                break
                    if found_swap:
                        break
                
                if not improved:
                    # Try 1-for-2: add one to A, one to B
                    valid_adds_B = []
                    for x in range(1, N + 1):
                        if x in used and x != a:
                            continue
                        if len(B) < MAX_EACH and is_valid_add(x, B, sumB, diffB, new_diffA):
                            valid_adds_B.append(x)
                    
                    for x1 in valid_adds_A:
                        if len(new_A) < MAX_EACH:
                            # Check if x1 can be added to A
                            # (Already checked in valid_adds_A generation, but need to ensure it doesn't conflict with B's new addition? No, B's addition is separate)
                            # Wait, valid_adds_A was checked against new_diffA and diffB.
                            # So x1 is valid for A.
                            
                            for x2 in valid_adds_B:
                                # Check if x2 is valid for B, given that A has x1?
                                # The diff sets must be disjoint.
                                # new_diffA contains diffs of A\{a}.
                                # temp_diffA contains diffs of A\{a} U {x1}.
                                # x2 must have diffs disjoint from temp_diffA.
                                
                                # Recalculate temp_diffA for x1
                                temp_diffA = new_diffA
                                for y in new_A:
                                    temp_diffA |= (1 << abs(x1 - y))
                                
                                # Check x2 against B and temp_diffA
                                if (sumB >> (x2 + x2)) & 1: continue
                                conflict = False
                                for y in B:
                                    if (sumB >> (x2 + y)) & 1:
                                        conflict = True; break
                                    d = abs(x2 - y)
                                    if (diffB >> d) & 1:
                                        conflict = True; break
                                    if (temp_diffA >> d) & 1:
                                        conflict = True; break
                                if not conflict:
                                    A = new_A + [x1]
                                    # Update A bitmasks
                                    sumA = new_sumA | (1 << (x1 + x1))
                                    diffA = temp_diffA
                                    for y in new_A:
                                        sumA |= (1 << (x1 + y))
                                        
                                    B = B + [x2]
                                    # Update B bitmasks
                                    sumB = sumB | (1 << (x2 + x2))
                                    diffB = diffB
                                    for y in B[:-1]: # Exclude x2
                                        diffB |= (1 << abs(x2 - y))
                                    
                                    used = set(A) | set(B)
                                    improved = True
                                    break
                        if improved:
                            break
                
                if improved:
                    break
            
            # Try removing from B
            if not improved:
                for i in range(len(B)):
                    if time.time() - start_time > DEADLINE - 5:
                        break
                    b = B[i]
                    new_B, new_sumB, new_diffB = remove_element(b, B, sumB, diffB)
                    
                    valid_adds_B = []
                    for x in range(1, N + 1):
                        if x in used and x != b:
                            continue
                        if len(new_B) < MAX_EACH and is_valid_add(x, new_B, new_sumB, new_diffB, diffA):
                            valid_adds_B.append(x)
                    
                    # Try 1-for-2: add two to B
                    found_swap = False
                    for j in range(len(valid_adds_B)):
                        x1 = valid_adds_B[j]
                        temp_sumB = new_sumB | (1 << (x1 + x1))
                        temp_diffB = new_diffB
                        for y in new_B:
                            temp_sumB |= (1 << (x1 + y))
                            temp_diffB |= (1 << abs(x1 - y))
                        
                        for k in range(j + 1, len(valid_adds_B)):
                            x2 = valid_adds_B[k]
                            if (temp_sumB >> (x2 + x2)) & 1: continue
                            conflict = False
                            for y in new_B:
                                if (temp_sumB >> (x2 + y)) & 1:
                                    conflict = True; break
                                d = abs(x2 - y)
                                if (temp_diffB >> d) & 1:
                                    conflict = True; break
                                if (diffA >> d) & 1:
                                    conflict = True; break
                            if not conflict:
                                if (temp_sumB >> (x1 + x2)) & 1: conflict = True
                                else:
                                    d12 = abs(x1 - x2)
                                    if (temp_diffB >> d12) & 1: conflict = True
                                    elif (diffA >> d12) & 1: conflict = True
                                
                                if not conflict:
                                    B = new_B + [x1, x2]
                                    sumB = temp_sumB | (1 << (x2 + x2))
                                    diffB = temp_diffB
                                    for y in new_B:
                                        sumB |= (1 << (x2 + y))
                                        diffB |= (1 << abs(x2 - y))
                                    diffB |= (1 << abs(x1 - x2))
                                    
                                    A, sumA, diffA = A, sumA, diffA
                                    used = set(A) | set(B)
                                    improved = True
                                    found_swap = True
                                    break
                        if found_swap:
                            break
                    
                    if not improved:
                        # Try 1-for-2: add one to B, one to A
                        valid_adds_A = []
                        for x in range(1, N + 1):
                            if x in used and x != b:
                                continue
                            if len(A) < MAX_EACH and is_valid_add(x, A, sumA, diffA, new_diffB):
                                valid_adds_A.append(x)
                        
                        for x1 in valid_adds_B:
                            if len(new_B) < MAX_EACH:
                                temp_diffB = new_diffB
                                for y in new_B:
                                    temp_diffB |= (1 << abs(x1 - y))
                                
                                for x2 in valid_adds_A:
                                    if (sumA >> (x2 + x2)) & 1: continue
                                    conflict = False
                                    for y in A:
                                        if (sumA >> (x2 + y)) & 1:
                                            conflict = True; break
                                        d = abs(x2 - y)
                                        if (diffA >> d) & 1:
                                            conflict = True; break
                                        if (temp_diffB >> d) & 1:
                                            conflict = True; break
                                    if not conflict:
                                        B = new_B + [x1]
                                        sumB = new_sumB | (1 << (x1 + x1))
                                        diffB = temp_diffB
                                        for y in new_B:
                                            sumB |= (1 << (x1 + y))
                                            
                                        A = A + [x2]
                                        sumA = sumA | (1 << (x2 + x2))
                                        diffA = diffA
                                        for y in A[:-1]:
                                            diffA |= (1 << abs(x2 - y))
                                        
                                        used = set(A) | set(B)
                                        improved = True
                                        break
                            if improved:
                                break
                    
                    if improved:
                        break
        
        return A, B, sumA, diffA, sumB, diffB

    # Main Loop: ILS with local search
    runs = 0
    while time.time() - start_time < DEADLINE - 5:
        A, B, sumA, diffA, sumB, diffB = greedy_construct()
        size = len(A) + len(B)
        
        # Apply local search if we have time and this is competitive
        if time.time() - start_time < DEADLINE - 12 and size >= best_size:
            A, B, sumA, diffA, sumB, diffB = local_search(A, B, sumA, diffA, sumB, diffB)
            size = len(A) + len(B)
        
        if size > best_size:
            best_size = size
            best_A = A
            best_B = B
        
        runs += 1
        if best_size >= 128:
            break

    # Ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    return [best_A, best_B]

if __name__ == "__main__":
    # Quick test
    A, B = construct()
    print(f"Size A: {len(A)}, Size B: {len(B)}, Total: {len(A)+len(B)}")
    # Verify
    def verify(A, B):
        # Check Sidon
        def is_sidon(S):
            sums = set()
            for i in range(len(S)):
                for j in range(i, len(S)):
                    s = S[i] + S[j]
                    if s in sums:
                        return False
                    sums.add(s)
            return True
        
        if not is_sidon(A) or not is_sidon(B):
            return False, "Sidon check failed"
            
        # Check Disjoint Diffs
        def get_diffs(S):
            return {abs(a-b) for i, a in enumerate(S) for b in S[i+1:]}
        
        dA = get_diffs(A)
        dB = get_diffs(B)
        
        if dA & dB:
            return False, f"Diff intersection: {dA & dB}"
            
        return True, "Valid"

    valid, msg = verify(A, B)
    print(f"Valid: {valid}, Msg: {msg}")
