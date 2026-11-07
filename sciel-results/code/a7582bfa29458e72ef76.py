import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 128
    
    # Bitset representation: bit i is 1 if i is in S.
    # We use 0-indexed bits for 1..128, so bit 0 is 1, bit 1 is 2, ..., bit 127 is 128.
    # Actually, let's use bit i for value i+1? No, let's use bit i for value i.
    # Value 1 -> bit 1? Or bit 0?
    # Let's use bit index = value. So bit 1 is value 1, bit 128 is value 128.
    # Python ints are arbitrary precision, so this is fine.
    
    def is_valid_bitset(S_bits):
        """
        Check if the set represented by S_bits is valid.
        S_bits is an int where bit i is 1 if i is in the set.
        """
        # Precompute sum_in_S for all i? 
        # sum_in_S[i] is a bitset of j such that i+j is in S_bits.
        # We can compute this on the fly or precompute for the current S.
        # Since S changes, we need to recompute or update.
        # For a full check, we can just iterate pairs.
        
        # Optimization: 
        # For each a in S:
        #   For each b in S with b > a:
        #     if a+b in S:
        #       # Check if there is c in S, c != a, c != b, such that a+c in S and b+c in S
        #       # This is equivalent to: (S & sum_in_S[a] & sum_in_S[b]) has any bit c != a, b.
        
        # Let's compute sum_in_S array for the current S_bits.
        # sum_in_S[i] = {j | i+j in S}
        # We can compute this efficiently.
        
        # Actually, simpler check without precomputing sum_in_S for all i:
        # Iterate a in S:
        #   Iterate b in S, b > a:
        #     s = a+b
        #     if s in S:
        #       # We need to check if there is c in S \ {a,b} such that a+c in S and b+c in S.
        #       # Let's iterate c in S, c > b? No, c can be anywhere.
        #       # But note: if a+c in S and b+c in S, then c is in (S - a) intersect (S - b).
        #       # (S - a) is the set of x such that x+a in S.
        
        # Let's stick to the bitset intersection idea but compute the "shifted" sets.
        # S_shifted_a = {x | x+a in S} = (S_bits >> a)
        # S_shifted_b = {x | x+b in S} = (S_bits >> b)
        # Common candidates for c are in S_bits & (S_bits >> a) & (S_bits >> b).
        # We must exclude a and b from c.
        # So check if (S_bits & (S_bits >> a) & (S_bits >> b) & ~(1<<a) & ~(1<<b)) != 0.
        
        # Wait, S_bits >> a gives bits j such that j+a is in S. So j is the value c.
        # Yes.
        
        # To iterate over elements in S, we can extract them.
        # But doing this for all pairs is O(|S|^2 * bit_op_cost).
        # Bit op cost is small (128 bits).
        
        # Let's extract elements of S into a list for iteration.
        S_list = []
        temp = S_bits
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            S_list.append(idx)
            temp ^= lsb
        
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i+1, n):
                b = S_list[j]
                s = a + b
                if s > N:
                    continue
                if not (S_bits & (1 << s)):
                    continue
                # Check for c
                # Candidates for c are in S, and c+a in S, and c+b in S.
                # c is in S_bits.
                # c+a in S => bit (c+a) is set in S_bits => c is in (S_bits >> a).
                # c+b in S => c is in (S_bits >> b).
                # So c is in S_bits & (S_bits >> a) & (S_bits >> b).
                candidates = S_bits & (S_bits >> a) & (S_bits >> b)
                # Exclude a and b
                candidates &= ~(1 << a)
                candidates &= ~(1 << b)
                if candidates:
                    return False
        return True

    def get_conflicts(S_bits, x):
        """
        Return list of elements y in S such that removing y allows adding x.
        Actually, we want to find the "blocking" sets.
        x cannot be added if there exist a,b in S such that a+b in S, a+x in S, b+x in S.
        For each such triple (a,b,x), the conflict is that {a,b,a+b,a+x,b+x} are all in S (except x which we want to add).
        Wait, if we add x, we form the forbidden config if a,b,a+b,a+x,b+x are in S.
        So to add x, we must remove at least one of {a, b, a+b, a+x, b+x} from S.
        Note: a+x and b+x are not in S yet (since x is not in S). So they are not "in S".
        Wait! The condition is: a,b,c in S AND a+b, a+c, b+c in S.
        If we add x to S, then x becomes c (or a or b).
        Case 1: x is c. Then we need a,b in S, a+b in S, a+x in S, b+x in S.
                But a+x and b+x are NOT in S (since x is new). So this case is impossible?
                No, a+x is a value. Is a+x in S?
                The set S currently does NOT contain x.
                Does S contain a+x? Yes, possibly.
                So if a,b in S, a+b in S, and a+x in S and b+x in S, then adding x creates a forbidden config with c=x.
                To prevent this, we must remove a, or b, or a+b, or a+x, or b+x.
                But a+x and b+x are in S. So removing them is an option.
        
        Case 2: x is a. Then we need b,c in S, x+b in S, x+c in S, b+c in S.
                Here x is the new element.
                So if b,c in S, b+c in S, and x+b in S and x+c in S, then adding x creates a forbidden config with a=x.
                To prevent this, remove b, c, b+c, x+b, or x+c.
        
        Case 3: x is b. Symmetric to Case 2.
        
        So, to add x, we must identify all "threats".
        A threat is a pair (u,v) in S such that u+v in S, and u+x in S, and v+x in S.
        Then adding x creates a forbidden config (x,u,v).
        We must remove one of {u, v, u+v, u+x, v+x}.
        
        Let's find all such threats.
        """
        threats = []
        S_list = []
        temp = S_bits
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            S_list.append(idx)
            temp ^= lsb
        
        # Check if x can be added safely.
        # Iterate all pairs (u,v) in S.
        n = len(S_list)
        for i in range(n):
            u = S_list[i]
            for j in range(i+1, n):
                v = S_list[j]
                s_uv = u + v
                if s_uv > N: continue
                if not (S_bits & (1 << s_uv)): continue
                # Check if u+x in S and v+x in S
                sx_u = u + x
                sx_v = v + x
                if sx_u > N or sx_v > N: continue
                if (S_bits & (1 << sx_u)) and (S_bits & (1 << sx_v)):
                    # Threat found: (u, v)
                    # Blocking set: {u, v, s_uv, sx_u, sx_v}
                    # We need to remove one of these.
                    threats.append((u, v, s_uv, sx_u, sx_v))
        
        return threats

    # Start with upper half
    base_bits = 0
    for i in range(65, 129):
        base_bits |= (1 << i)
    
    best_bits = base_bits
    best_size = 64
    
    current_bits = base_bits
    
    # Phase 1: Greedy Add
    for x in range(1, 129):
        if time.time() > deadline: break
        if current_bits & (1 << x): continue
        trial = current_bits | (1 << x)
        if is_valid_bitset(trial):
            current_bits = trial
            if bin(current_bits).count('1') > best_size:
                best_size = bin(current_bits).count('1')
                best_bits = current_bits

    # Phase 2: Conflict-Driven Local Search
    # Try to add a random x. If blocked, try removing a blocker.
    if time.time() < deadline:
        for step in range(10000):
            if time.time() > deadline: break
            
            # Pick a random x not in current
            candidates_to_add = [i for i in range(1, 129) if not (current_bits & (1 << i))]
            if not candidates_to_add:
                break
            x = random.choice(candidates_to_add)
            
            threats = get_conflicts(current_bits, x)
            
            if not threats:
                # Can add x
                current_bits |= (1 << x)
                size = bin(current_bits).count('1')
                if size > best_size:
                    best_size = size
                    best_bits = current_bits
            else:
                # Pick a random threat
                u, v, s_uv, sx_u, sx_v = random.choice(threats)
                # Blockers: u, v, s_uv, sx_u, sx_v
                # We want to remove one and add x.
                # Removing one might allow x to be added.
                # Let's try removing each blocker and checking if x can be added.
                # To save time, pick one random blocker.
                blocker = random.choice([u, v, s_uv, sx_u, sx_v])
                
                # Remove blocker
                new_bits = current_bits & ~(1 << blocker)
                # Add x
                new_bits |= (1 << x)
                
                # Check validity
                if is_valid_bitset(new_bits):
                    # Accept if size doesn't decrease (swap) or increases
                    # Here we removed 1 and added 1, so size is same.
                    # We might improve structure for future additions.
                    # Accept with some probability or always?
                    # Let's accept if it allows more additions later.
                    # For now, accept if valid.
                    current_bits = new_bits
                    size = bin(current_bits).count('1')
                    if size > best_size:
                        best_size = size
                        best_bits = current_bits

    # Phase 3: Random Restart / Perturbation
    if time.time() < deadline:
        for restart in range(20):
            if time.time() > deadline: break
            
            # Perturb best_bits
            trial_bits = best_bits
            # Remove 2-4 random elements
            elems = []
            temp = trial_bits
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                elems.append(idx)
                temp ^= lsb
            
            to_remove = random.sample(elems, min(4, len(elems)))
            for e in to_remove:
                trial_bits &= ~(1 << e)
            
            # Greedy Add
            for x in range(1, 129):
                if time.time() > deadline: break
                if trial_bits & (1 << x): continue
                t2 = trial_bits | (1 << x)
                if is_valid_bitset(t2):
                    trial_bits = t2
            
            size = bin(trial_bits).count('1')
            if size > best_size:
                best_size = size
                best_bits = trial_bits

    # Convert best_bits to list
    result = []
    temp = best_bits
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        result.append(idx)
        temp ^= lsb
        
    return sorted(result)
