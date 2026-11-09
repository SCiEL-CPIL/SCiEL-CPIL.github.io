def find_root(f, x0):
    x = float(x0)
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    h = 1e-4 * max(1.0, abs(x))
    x1 = x + h
    fx1 = f(x1)
    if abs(fx1) <= 1e-8:
        return x1
    
    a, b, fa, fb = None, None, None, None
    best_x, best_fx = x, fx
    
    for i in range(490):
        if abs(fx) < abs(best_fx):
            best_x, best_fx = x, fx
        if abs(fx1) < abs(best_fx):
            best_x, best_fx = x1, fx1
        
        if abs(best_fx) <= 1e-8:
            return best_x
        
        if a is not None and b is not None:
            mid = (a + b) / 2.0
            fmid = f(mid)
            if abs(fmid) <= 1e-8:
                return mid
            if abs(fmid) < abs(best_fx):
                best_x, best_fx = mid, fmid
            if fa * fmid < 0:
                b, fb = mid, fmid
            else:
                a, fa = mid, fmid
        else:
            if fx1 == fx:
                x1 = x + 2*h
                fx1 = f(x1)
                if abs(fx1) <= 1e-8:
                    return x1
                continue
            
            x2 = x1 - fx1 * (x1 - x) / (fx1 - fx)
            fx2 = f(x2)
            if abs(fx2) <= 1e-8:
                return x2
            if abs(fx2) < abs(best_fx):
                best_x, best_fx = x2, fx2
            
            if fx * fx2 < 0:
                a, b, fa, fb = x, x2, fx, fx2
            elif fx1 * fx2 < 0:
                a, b, fa, fb = x1, x2, fx1, fx2
            
            x, fx = x1, fx1
            x1, fx1 = x2, fx2
    
    return best_x
