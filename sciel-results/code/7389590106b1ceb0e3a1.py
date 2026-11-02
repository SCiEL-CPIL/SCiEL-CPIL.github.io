    def decompose(n, m, p):
        # Base cases
        if n == 1 and m == 1 and p == 1:
            U = np.ones((1, 1), dtype=np.int64)
            V = np.ones((1, 1), dtype=np.int64)
            W = np.ones((1, 1), dtype=np.int64)
            return {'U': U, 'V': V, 'W': W}
        
        if n == 2 and m == 2 and p == 2:
            return strassen_222()
            
        # If any dimension is 1, use schoolbook (no reduction possible via standard splits)
        if n == 1 or m == 1 or p == 1:
            return schoolbook(n, m, p)
            
        # Choose the dimension to split. We split the largest dimension to balance the recursion.
        # Or we can try all splits and pick the one with minimal rank.
        # For simplicity and speed, we will split the largest dimension into 1 and rest.
        # This allows 2x2x2 Strassen to be used if a 2 appears.
        
        candidates = []
        
        # Try splitting n
        if n > 1:
            n1, n2 = 1, n - 1
            d1 = decompose(n1, m, p)
            d2 = decompose(n2, m, p)
            
            r1 = d1['U'].shape[1]
            r2 = d2['U'].shape[1]
            r_total = r1 + r2
            
            U = np.zeros((n*m, r_total), dtype=np.int64)
            V = np.zeros((m*p, r_total), dtype=np.int64)
            W = np.zeros((p*n, r_total), dtype=np.int64)
            
            # U: Rows are (n1*m) and (n2*m)
            U[:n1*m, :r1] = d1['U']
            U[n1*m:, r1:] = d2['U']
            
            # V: Rows are m*p (same for both), Cols are r1 and r2
            V[:, :r1] = d1['V']
            V[:, r1:] = d2['V']
            
            # W: Rows are (p*n1) and (p*n2)
            W[:p*n1, :r1] = d1['W']
            W[p*n1:, r1:] = d2['W']
            
            candidates.append({'U': U, 'V': V, 'W': W, 'rank': r_total})
            
        # Try splitting m
        if m > 1:
            m1, m2 = 1, m - 1
            d1 = decompose(n, m1, p)
            d2 = decompose(n, m2, p)
            
            r1 = d1['U'].shape[1]
            r2 = d2['U'].shape[1]
            r_total = r1 + r2
            
            U = np.zeros((n*m, r_total), dtype=np.int64)
            V = np.zeros((m*p, r_total), dtype=np.int64)
            W = np.zeros((p*n, r_total), dtype=np.int64)
            
            # U: Rows are (n*m1) and (n*m2)
            U[:n*m1, :r1] = d1['U']
            U[n*m1:, r1:] = d2['U']
            
            # V: Rows are (m1*p) and (m2*p)
            V[:m1*p, :r1] = d1['V']
            V[m1*p:, r1:] = d2['V']
            
            # W: Rows are p*n (same for both), Cols are r1 and r2
            W[:, :r1] = d1['W']
            W[:, r1:] = d2['W']
            
            candidates.append({'U': U, 'V': V, 'W': W, 'rank': r_total})
            
        # Try splitting p
        if p > 1:
            p1, p2 = 1, p - 1
            d1 = decompose(n, m, p1)
            d2 = decompose(n, m, p2)
            
            r1 = d1['U'].shape[1]
            r2 = d2['U'].shape[1]
            r_total = r1 + r2
            
            U = np.zeros((n*m, r_total), dtype=np.int64)
            V = np.zeros((m*p, r_total), dtype=np.int64)
            W = np.zeros((p*n, r_total), dtype=np.int64)
            
            # U: Rows are n*m (same for both), Cols are r1 and r2
            U[:, :r1] = d1['U']
            U[:, r1:] = d2['U']
            
            # V: Rows are (m*p1) and (m*p2)
            V[:m*p1, :r1] = d1['V']
            V[m*p1:, r1:] = d2['V']
            
            # W: Rows are (p1*n) and (p2*n)
            W[:p1*n, :r1] = d1['W']
            W[p1*n:, r1:] = d2['W']
            
            candidates.append({'U': U, 'V': V, 'W': W, 'rank': r_total})
            
        if not candidates:
            return schoolbook(n, m, p)
            
        # Pick the candidate with the minimum rank
        best = min(candidates, key=lambda x: x['rank'])
        return {'U': best['U'], 'V': best['V'], 'W': best['W']}
