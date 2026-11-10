def construct():
    p = 1009
    # Generate the set of all quadratic residues modulo p
    # x^2 mod p for x in 1..p-1
    qr_set = set()
    for x in range(1, p):
        qr_set.add((x * x) % p)
        
    # Find the smallest s such that s, s+1, s+2, s+3, s+4 are all in qr_set
    for s in range(1, 1005):
        if (s in qr_set and 
            (s+1) in qr_set and 
            (s+2) in qr_set and 
            (s+3) in qr_set and 
            (s+4) in qr_set):
            
            # Find a root for each of the 5 residues
            roots = []
            for i in range(5):
                target = s + i
                # Find x such that x^2 % p == target
                # x must be in 1..p-1
                # We can search up to (p-1)//2 because if x is a root, p-x is also a root.
                # We just need one valid root.
                root = None
                for x in range(1, (p // 2) + 1):
                    if (x * x) % p == target:
                        root = x
                        break
                if root is None:
                    # Fallback: search full range (should not happen)
                    for x in range(1, p):
                        if (x * x) % p == target:
                            root = x
                            break
                if root is None:
                    raise RuntimeError(f"Failed to find root for {target}")
                roots.append(root)
                
            return {'start': s, 'roots': roots}
            
    raise RuntimeError('No valid run found')
