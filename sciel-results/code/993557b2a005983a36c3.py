def _is_convex(q):
    n = len(q)
    diff = q[:,None,:] - q[None,:,:]
    pd = np.sqrt((diff**2).sum(axis=2))
    np.fill_diagonal(pd, np.inf)
    if pd.min() < 1e-8:
        return False
    total = 0.0
    prev = 0
    for i in range(n):
        a=q[i]; b=q[(i+1)%n]; c=q[(i+2)%n]
        ab=b-a; bc=c-b
        cross=ab[0]*bc[1]-ab[1]*bc[0]
        if abs(cross) < 1e-10: return False
        s = 1 if cross>0 else -1
        if prev==0: prev=s
        elif s!=prev: return False
        a1=np.arctan2(ab[1],ab[0]); a2=np.arctan2(bc[1],bc[0])
        turn=(a2-a1+np.pi)%(2*np.pi)-np.pi
        total+=turn
    return abs(abs(total)-2*np.pi) < 1e-4
