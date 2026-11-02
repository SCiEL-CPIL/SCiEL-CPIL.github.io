def add_dict(d1, d2, coeff1=1, coeff2=1):
    result = {}
    for k, v in d1.items():
        result[k] = result.get(k, 0) + coeff1*v
    for k, v in d2.items():
        result[k] = result.get(k, 0) + coeff2*v
    return result
