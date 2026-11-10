_last = {}
def _obj(x):
    a,b = _recon(x)
    v = _error_from_coeffs(a,b)
    _last['x']=x; _last['a']=a; _last['b']=b; _last['v']=v
    if v < best['v']:
        best['a']=a.tolist(); best['b']=b.tolist(); best['v']=v
    return v
def cb(x):
    if time.time() > deadline:
        raise TimeoutError
    return False
